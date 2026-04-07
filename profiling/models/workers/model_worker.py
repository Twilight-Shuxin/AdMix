import copy
import os
import sys
import time
import ctypes
import queue
import threading
import torch
import inspect
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.multiprocessing import Process
from profiling.constants import *
from trace.trace_utils import SimpleTimeTrace
from profiling.models.workers.constants import SyncType

# ── CUDA Driver API helpers (MPS context management) ─────────────────────────

_libcuda = None

def _cuda_driver():
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda

def _cu(result):
    if result != 0:
        raise RuntimeError(f"CUDA driver error code {result}")

def _create_cuda_context(device_id, thread_pct):
    """Create a CUDA driver context with the given MPS thread percentage.
    The env var is read at cuCtxCreate time, so must be set beforehand.
    The context is popped so it can be pushed later per-thread via switch_context."""
    lib = _cuda_driver()
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
    cu_device = ctypes.c_int()
    _cu(lib.cuDeviceGet(ctypes.byref(cu_device), device_id))
    ctx = ctypes.c_void_p()
    _cu(lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, cu_device))
    _cu(lib.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p())))
    return ctx

def _switch_context(ctx):
    """Make ctx the active CUDA context for the calling thread."""
    _cu(_cuda_driver().cuCtxSetCurrent(ctx))

_ctx_create_lock = threading.Lock()  # serialises cuCtxCreate_v2 calls (env-var race)
_pool_op_lock    = threading.Lock()  # serialises use_mem_pool contexts and pool destruction
                                     # (captures_underway is global per-device, so concurrent
                                     # use_mem_pool + MemPool.__del__ across threads triggers
                                     # the captures_underway.empty() assert in PyTorch)

def _resp_get(resp_q):
    """Get a response from a worker thread's resp_q, re-raising if the thread crashed."""
    result = resp_q.get()
    if isinstance(result, Exception):
        raise RuntimeError("mps context worker thread crashed") from result
    return result

class _Cmd:
    """Control command sent to a context worker thread."""
    __slots__ = ("kind", "payload", "resp_q")
    def __init__(self, kind, payload=None):
        self.kind    = kind
        self.payload = payload
        self.resp_q  = queue.Queue()      # receives result when the cmd completes

def _pin_model(model):
    """Move all model parameters and buffers into pinned (page-locked) CPU memory
    so DMA transfers to GPU avoid an intermediate CPU staging copy."""
    for param in model.parameters():
        param.data = param.data.pin_memory()
    for buf in model.buffers():
        buf.data = buf.data.pin_memory()
    return model

def _load_model_to_gpu(model_shell, model_pinned, device, model_pool):
    """Allocate an empty GPU shell from the pre-built meta shell (no H2D data copy),
    then stream weights directly from pinned CPU memory via DMA (non_blocking=True).
    Caller must pass a freshly created model_pool. Weights are allocated into it.
    use_mem_pool is held under _pool_op_lock; sync runs outside to avoid blocking others."""
    with _pool_op_lock:
        with torch.cuda.memory.use_mem_pool(model_pool):
            gpu_model = copy.deepcopy(model_shell).to_empty(device=device)
            for gp, cp in zip(gpu_model.parameters(), model_pinned.parameters()):
                gp.data.copy_(cp.data, non_blocking=True)
            for gb, cb in zip(gpu_model.buffers(), model_pinned.buffers()):
                gb.data.copy_(cb.data, non_blocking=True)
    torch.cuda.synchronize(device)   # wait for DMA outside lock; copies already queued
    return gpu_model

# ─────────────────────────────────────────────────────────────────────────────

class ModelWorker(Process):
    def __init__(self, model_name, devices, pipe, worker_barrier, single_worker_barrier, log_path, profile_path):
        self.model_name = model_name
        self.pipe = pipe
        self.worker_barrier = worker_barrier
        self.single_worker_barrier = single_worker_barrier
        self.log_path = log_path
        self.profile_path = profile_path
        self.device = "cuda"
        self.devices = devices
        os.environ["CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING"] = "1"
        _cu(_cuda_driver().cuInit(0))
        super().__init__()
    
    def init_start(self):
        return
    
    def shutdown(self):
        return
    
    """
        Setup the parameters related to submodules.
        FIXME: Currently ignored all partial model setup due to memory restrictions.
    """
    def submodules_setup(self):
        self.submodules = None
        self.total_module = -1
        if hasattr(self, "model") and hasattr(self.model, "get_submodules") and callable(self.model.get_submodules):
            self.submodules = self.model.get_submodules()
            self.total_module = len(self.submodules)
        # self.partial_model = {}
        # self.partial_input = {}
        # for batch_size in self.supported_batchsize:
        #     self.partial_input[batch_size] = {}
        #     self.partial_input[batch_size][0] = self.random_input[batch_size]
        # for i in range(0, self.total_module, PARTIAL_MODEL_STRIDE):
        #     right = min(i+PARTIAL_MODEL_STRIDE, self.total_module)
        #     self.partial_model[i] = torch.nn.Sequential(*self.submodules[i:right]).cuda(self.device)
        #     if(right == self.total_module):
        #         break
        #     for batch_size in self.supported_batchsize:
        #         partial_input_current = self.partial_input[batch_size][i]
        #         self.partial_input[batch_size][right] = self._inference(partial_input_current, i, right)
    
    @classmethod
    def get_input_config_from_test_config(cls, test_config):
        return InputConfig(batch_size = test_config.batch_size, nosync = test_config.nosync, start = test_config.start, 
            inter_flag = test_config.inter_flag) 
    
    def get_input(self, input_config):
        batch_size = input_config.batch_size
        if(input_config.nosync):
            batch_size = 1
        if input_config.inter_flag:
            return self.partial_input[batch_size][input_config.start]
        else:
            return self.random_input[batch_size]

    """
        Profile model to get the 99 percentile latency of each batch size setting.
        Return profiled data to master.
        NOTE: 
    """
    def profile(self):
        def _warmup(batch_size, warmup_iter = 5):
            for _ in range(warmup_iter):
                self.inference(self.random_input[batch_size], sync = True)
            return

        def _profile(batch_size, test_time):
            profile_iter = 15
            spent_time = []
            iter_cnt = 0
            profile_start = time.perf_counter()
            while time.perf_counter() - profile_start < test_time or iter_cnt < profile_iter:
                start_time = time.perf_counter()
                self.inference(self.random_input[batch_size], sync = True)
                spent_time.append(time.perf_counter() - start_time)
                iter_cnt += 1
            return spent_time

        profile_data = {}
        dcgm_group_id = create_dcgm_group(self.devices)
        for batch_size in tqdm(self.supported_batchsize[::-1], disable=not sys.stderr.isatty()):
            _warmup(batch_size)
            dcgm_monitor_conn = start_dcgm_single('sm_active', self.devices, dcgm_group_id = dcgm_group_id)
            spent_time = _profile(batch_size = batch_size, test_time = 5)
            dcgm_records = parse_dcgm_output(stop_dcgm(dcgm_monitor_conn, del_dcgm_group_flag = False)[0])
            active_dcgm = [v for _, v in dcgm_records[1:] if v != 0]
            profile_data[batch_size] = {
                'latency': np.percentile(spent_time, 99),
                'sm_active': round((sum(active_dcgm) / len(active_dcgm)) * 100, 2) if active_dcgm else 0,
            }
        delete_dcgm_group(dcgm_group_id)
        _warmup(self.profile_pw_bs)
        power_monitor_conn = start_power_monitor(self.devices)
        iter_cnt = len(_profile(batch_size = self.profile_pw_bs, test_time = 30))
        power_records = parse_power_output(stop_power_monitor(power_monitor_conn))
        profile_data[self.profile_pw_bs]['avg_energy'] = sum([v for _, v in power_records]) / (iter_cnt * self.profile_pw_bs)
        self.pipe.send(dict(sorted(profile_data.items())))
    
     # Setup submodel under current test_configuration: self.submodel = model[start:end]
    def setup_test_submodel(self):
        if(self.test_config.inter_flag):
            self.submodel = torch.nn.Sequential(*self.submodules[self.test_config.start : self.test_config.end])
    
    # Get the trace according to trace path
    def get_trace(self):
        if(self.test_config.trace_path):
            self.test_config.trace = SimpleTimeTrace(self.test_config.trace_path)    

    def setup_testenv(self, test_config):
        self.test_config = test_config
        self.setup_test_submodel()
        self.get_trace()
        self.input_config = self.get_input_config_from_test_config(self.test_config)
    
    def init_profile_data(self):
        self.profile_data = ProfileData()
        if(self.test_config.recording_action.record_statistics):
            self.profile_data.logs = []
            self.profile_data.absolute_throughput_records = TimeBinRecord(init = 0, time_limit = self.test_config.test_time)
    
    def init_runtime_data(self):
        self.runtime_data = RuntimeData()
        self.runtime_data.absolute_throughput = 0

    # NOTE: Only llm_worker is actually engaged with scheduling data; 
    #   Other workers only react to corresponding client's scheduled tasks.
    def init_scheduler_data(self):
        self.scheduler_data = SchedulerData()
    
    def init_testenv_data(self):
        self.init_profile_data()
        self.init_runtime_data()
        self.init_scheduler_data()
    
    def inference(self, input, start = None, end = None, sync = True):
        if((start is not None) and (end is not None) and (start != 0 or end != self.total_module)): 
            result = input
            for i in range(start, end, PARTIAL_MODEL_STRIDE):
                result = self.partial_model[i](result)
        else:
            result = self.model(input)
        if(sync): 
            torch.cuda.synchronize()
        return result
    
    """
        Perform inferences for GPU & model warmup.
    """
    def warmup(self):
        warmup_iter = 10
        model_input = self.get_input(self.input_config)
        for _ in range(warmup_iter):
            self.inference(model_input, self.test_config.start, self.test_config.end, sync = True)
    
    """
        Perform test according to self.test_config.
        If scheduler is set, use scheduler's serve method instead.
        All runtime settings have been completed by the wrapper function 'test' ↓
    """
    def _test(self):
        self.init_testenv_data()
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time
        model_input = self.get_input(self.input_config)
        while(time.perf_counter() < self.runtime_data.time_limit):
            if(self.test_config.trace_path):
                req_time = self.test_config.trace.next()
                sleep_time = req_time + self.runtime_data.start_time_pointer - time.perf_counter()
                if(sleep_time > 0):
                    time.sleep(sleep_time)
                if(time.perf_counter() > self.runtime_data.time_limit):
                    break

            req_start_time = time.perf_counter()
            if(self.test_config.nosync):
                for i in range(self.test_config.batch_size):
                    self.inference(model_input, self.test_config.start, self.test_config.end, sync=False)
                torch.cuda.synchronize()
            else:
                self.inference(model_input, self.test_config.start, self.test_config.end, sync=True)
            req_process_time = time.perf_counter() - req_start_time

            self.runtime_data.absolute_throughput += self.test_config.batch_size
            self.profile_data.absolute_throughput_records.add(time.perf_counter() - self.runtime_data.start_time_pointer, self.test_config.batch_size)
            if(self.test_config.recording_action.record_statistics):
                self.profile_data.logs.append((self.test_config.batch_size, req_process_time * 1000))
        # End test

    """
        Perform test according to self.test_config, supports sequential test, trace test and scheduler test. 
        Initializes the runtime settings for test and execute test.
    """
    def test(self):
        if inspect.iscoroutinefunction(self._test):
            self.loop.run_until_complete(self._test())
        else:
            self._test()
    
    def send_performance_data(self):
        self.pipe.send(self.runtime_data.absolute_throughput)
    
    def send_profiled_data(self):
        self.pipe.send(self.profile_data.logs)
        self.pipe.send(self.profile_data.absolute_throughput_records)
    
    def context_setup(self, device_id=0):
        """Initialise the MPS context infrastructure."""
        self.model = self.model.to('cpu')
        self._model_shell = copy.deepcopy(self.model).to('meta')  # reusable empty shell (no data)
        self.model = _pin_model(self.model)   # page-lock for fast DMA on every H2D copy
        self.random_input = {bs: t.cpu() for bs, t in self.random_input.items()} 
        torch.cuda.empty_cache()

        self._ctx_device_id = device_id
        self._mps_ctx_lock      = threading.Lock()  # protects ctx_* dicts
        self.MPS_step       = 5                 # neighbour pre-create distance
        self.mps_batch_size_map      = defaultdict(int)               # mps -> max confirmed valid batch size
        self.mps_batch_size_exceeded = defaultdict(lambda: float('inf'))  # mps -> min confirmed exceeded batch size

        # Per-context state: 'created' | 'loaded' | 'active' | 'pending_unload' | 'unloading'
        self.ctx_handles             = {}   # mps → ctypes.c_void_p  (CUDA driver context)
        self.ctx_threads             = {}   # mps → Thread
        self.ctx_queues              = {}   # mps → Queue  (inference work + _Cmd control items)
        self.ctx_states              = {}   # mps → str
        self.ctx_unloading_events = {}   # mps → Event  (set when unload completes)
        self.ctx_deact_timers        = {}   # mps → threading.Timer  (pending delayed unload)
        self.unload_delay      = 5.0  # seconds to wait before actually unloading GPU model
        self.current_ctx_pct = None

        self._ctx_create(10)
        self._ctx_create(100)

    def _ctx_create(self, mps):
        """Create a CUDA driver context for `mps` and start its worker thread.
        The model is NOT loaded — call _load(mps) separately.
        No-op if the context already exists."""
        with self._mps_ctx_lock:
            if mps in self.ctx_states:
                return
        device_id  = self._ctx_device_id
        device     = f"cuda:{device_id}"
        with _ctx_create_lock:
            ctx = _create_cuda_context(device_id=device_id, thread_pct=mps)
        work_queue = queue.Queue()
        with self._mps_ctx_lock:
            self.ctx_handles[mps] = ctx
            self.ctx_queues[mps]  = work_queue
            self.ctx_states[mps]  = 'created'

        def _target(ctx=ctx, work_queue=work_queue):
            lib = _cuda_driver()
            _switch_context(ctx)        # bind CUDA context to this thread once
            torch.set_grad_enabled(False)
            gpu_model      = None
            model_pool     = None       # MemPool owning model weights
            loop_input     = None       # if set, keep inferring this input
            loop_bs        = 0
            _current_cmd   = None       # track the in-flight command for error reporting
            alive = True
            try:
                while alive:
                    # In loop mode: run one inference then check for new commands non-blocking.
                    # In idle mode: block until a command arrives.
                    if loop_input is not None:
                        t0 = time.perf_counter()
                        gpu_model(loop_input)
                        torch.cuda.synchronize()
                        elapsed = round(time.perf_counter() - t0, 3) * 1000
                        self.runtime_data.absolute_throughput += loop_bs
                        if self.test_config.recording_action.record_statistics:
                            self.profile_data.logs.append((loop_bs, elapsed))
                            self.profile_data.absolute_throughput_records.add(
                                time.perf_counter() - self.runtime_data.start_time_pointer, loop_bs)
                        if(self.mps_batch_size_exceeded[mps] > self.mps_batch_size_map[mps] + 1):
                            if elapsed < 110.0:
                                self.mps_batch_size_map[mps] = max(self.mps_batch_size_map[mps], loop_bs)
                            else:
                                self.mps_batch_size_exceeded[mps] = min(self.mps_batch_size_exceeded[mps], loop_bs)
                            exceeded = self.mps_batch_size_exceeded[mps]
                            if exceeded == float('inf'):
                                max_bs = max(self.supported_batchsize)
                                loop_bs = min(loop_bs * 2, max_bs)
                                if loop_bs == max_bs and elapsed < 110.0:
                                    self.mps_batch_size_exceeded[mps] = max_bs + 1  # all supported sizes are valid; cap exceeded
                            else:
                                loop_bs = (self.mps_batch_size_map[mps] + int(exceeded)) // 2
                            loop_input = self.get_input(InputConfig(batch_size=loop_bs)).to(device, copy=True)
                        elif(loop_bs != self.mps_batch_size_map[mps]):
                            loop_bs = self.mps_batch_size_map[mps]
                            loop_input = self.get_input(InputConfig(batch_size=loop_bs)).to(device, copy=True)
                        try:
                            item = work_queue.get_nowait()
                        except queue.Empty:
                            continue
                    else:
                        item = work_queue.get()
                    if isinstance(item, _Cmd):
                        _current_cmd = item
                        t0 = time.perf_counter()
                        if item.kind == 'load':
                            model_pool = torch.cuda.memory.MemPool()
                            gpu_model = _load_model_to_gpu(self._model_shell, self.model, device, model_pool)
                        elif item.kind == 'loop':
                            loop_bs = item.payload
                            if(loop_bs == 0):
                                loop_input = None # input is None but thread state remains active;
                            else:
                                if(self.mps_batch_size_map[mps]):
                                    loop_bs = self.mps_batch_size_map[mps]
                                loop_input = self.get_input(InputConfig(batch_size=loop_bs)).to(device, copy=True)
                            torch.cuda.synchronize()
                        elif item.kind == 'stop_loop':
                            loop_input = None
                            torch.cuda.empty_cache()
                        elif item.kind == 'unload':
                            gpu_model = None
                            with _pool_op_lock:   # no use_mem_pool active while pool is destroyed
                                model_pool = None
                        elif item.kind == 'destroy':
                            _cu(lib.cuCtxDestroy_v2(ctx))
                            alive = False
                        elapsed = round(time.perf_counter() - t0, 3) * 1000
                        item.resp_q.put(elapsed)
                        _current_cmd = None
            except Exception as exc:
                print(f"[mps-ctx-{mps}pct] FATAL exception: {exc}\n", flush=True)
                if _current_cmd is not None:
                    _current_cmd.resp_q.put(exc)
                raise

        t = threading.Thread(target=_target, name=f"mps-ctx-{mps}pct", daemon=True)
        t.start()
        with self._mps_ctx_lock:
            self.ctx_threads[mps] = t

    def _send_cmd_mps_thread(self, mps, kind, payload=None):
        """Send a control command to the `mps` worker thread and block until it completes."""
        cmd = _Cmd(kind, payload)
        self.ctx_queues[mps].put(cmd)
        return _resp_get(cmd.resp_q)

    def _get_active_mps(self):
        """Return the mps whose ctx_state is 'active', or None. Must be called under _mps_ctx_lock."""
        return next((p for p, s in self.ctx_states.items() if s == 'active'), None)

    def _load(self, mps):
        """Load model into GPU for `mps` context. Blocks until ready. Returns elapsed seconds.

        If a unload is still pending (timer not yet fired), cancels it atomically
        and returns immediately — the model is still on GPU so no H2D transfer is needed.
        If unload is already in flight, waits for it to finish before reloading."""
        # Atomically cancel any pending unload and reclaim the 'active' state.
        # Both this block and the timer callback (_on_unload_timer in _unload) acquire
        # self._mps_ctx_lock before touching ctx_states, so only one wins the transition.
        with self._mps_ctx_lock:
            timer = self.ctx_deact_timers.pop(mps, None)
            cancelled_pending = False
            if self.ctx_states.get(mps) == 'pending_unload': # Model is still on GPU — just flip back to loaded.
                self.ctx_states[mps] = 'loaded'
                cancelled_pending = True
            if timer is not None:
                timer.cancel()  # no-op if it already fired; state guard above handles that case

        if cancelled_pending:
            return 0.0

        with self._mps_ctx_lock: # Wait for any already-in-flight unload before reloading.
            evt = self.ctx_unloading_events.get(mps)
        if evt is not None:
            evt.wait()

        elapsed = self._send_cmd_mps_thread(mps, 'load')
        with self._mps_ctx_lock:
            self.ctx_states[mps] = 'loaded'
        return elapsed

    def start_inference_loop(self, mps, batch_size):
        """Instruct the `mps` thread to loop inference continuously until the next command."""
        self._send_cmd_mps_thread(mps, 'loop', payload=batch_size)
        with self._mps_ctx_lock:
            self.ctx_states[mps] = 'active'

    def _unload(self, mps, delay=None):
        """Schedule (or immediately execute) model unload for `mps`.

        delay > 0  (default self.unload_delay):
            Non-blocking. Sets state to 'pending_unload' and starts a timer.
            If _load(mps) is called before the timer fires, the unload is
            cancelled and the model stays loaded — no H2D reload needed.
        delay == 0:
            Blocking. Unloads the model immediately (used by _destroy_ctx / teardown).
        """
        if(mps is None):
            return
        if delay is None:
            delay = self.unload_delay

        def _do_unload(done_event):
            """Send unload to the worker thread and wait for completion."""
            self._send_cmd_mps_thread(mps, 'unload')
            if done_event is not None:
                done_event.set()
            with self._mps_ctx_lock:
                self.ctx_unloading_events.pop(mps, None)
                self.ctx_states[mps] = 'created'

        def _on_unload_timer():
            """Timer callback: stop the inference loop for `mps`. Model stays on GPU.
            The actual unload happens later via _unload(delay=0) / _destroy_ctx."""
            with self._mps_ctx_lock:
                # _load may have already flipped state back to 'loaded' — abort if so.
                if self.ctx_states.get(mps) != 'pending_unload':
                    return
                self.ctx_deact_timers.pop(mps, None)
                self.ctx_states[mps] = 'unloading'
                done_event = threading.Event()
                self.ctx_unloading_events[mps] = done_event
            _do_unload(done_event)


        # Cancel any previously scheduled timer for this context.
        with self._mps_ctx_lock:
            old_timer = self.ctx_deact_timers.pop(mps, None)
        if old_timer is not None:
            old_timer.cancel()

        if delay == 0:
            # Immediate path: atomically claim the unload slot.
            with self._mps_ctx_lock:
                state = self.ctx_states.get(mps)
                if state not in ('active', 'pending_unload', 'loaded'):
                    return  # not loaded, already unloading, or destoryed
                self.ctx_states[mps] = 'unloading'
                done_event = threading.Event()
                self.ctx_unloading_events[mps] = done_event
            _do_unload(done_event)
            return

        # Delayed path: schedule and record the timer.
        with self._mps_ctx_lock:
            self.ctx_states[mps] = 'pending_unload'
        self._send_cmd_mps_thread(mps, 'stop_loop')  # block until the worker finishes its current pass and stops looping
        timer = threading.Timer(delay, _on_unload_timer)
        with self._mps_ctx_lock:
            self.ctx_deact_timers[mps] = timer
        timer.start()

    def _destroy_ctx(self, mps):
        """unload (if active/pending/unloading) then destroy CUDA context for `mps`. Blocks until done."""
        # Cancel any pending timer first.
        with self._mps_ctx_lock:
            timer = self.ctx_deact_timers.pop(mps, None)
        if timer is not None:
            timer.cancel()
        # Force immediate unload. If the timer had fired (race), _unload(delay=0) sees 'unloading' and returns early — in-flight unload is caught below.
        self._unload(mps, delay=0)

        # Wait for any in-flight unload that we didn't initiate (timer race).
        with self._mps_ctx_lock:
            state = self.ctx_states.get(mps)
            evt   = self.ctx_unloading_events.get(mps)
        if state == 'unloading' and evt is not None:
            evt.wait()

        self._send_cmd_mps_thread(mps, 'destroy')
        self.ctx_threads[mps].join()
        with self._mps_ctx_lock:
            del self.ctx_handles[mps]
            del self.ctx_threads[mps]
            del self.ctx_queues[mps]
            del self.ctx_states[mps]

    def context_teardown(self):
        """Stop inference loop, unload, and destroy all MPS contexts."""
        with self._mps_ctx_lock:
            all_mps = list(self.ctx_states.keys())
            timers = {mps: self.ctx_deact_timers.pop(mps, None) for mps in all_mps}
        for timer in timers.values():
            if timer is not None:
                timer.cancel()
        with self._mps_ctx_lock:
            active_mps = self._get_active_mps()
        if active_mps is not None:
            self._send_cmd_mps_thread(active_mps, 'stop_loop')
            with self._mps_ctx_lock:
                self.ctx_states[active_mps] = 'loaded'
        for mps in all_mps:
            self._destroy_ctx(mps)
        self.current_ctx_pct = None

    def context_switch(self, mps):
        """Switch to `mps` context.

        Phase 1 (parallel): activate new context + unload old context simultaneously.
        Phase 2 (background): pre-create neighbor contexts (mps ± MPS_step)

        Terminology:
          create/destroy  — cuCtxCreate_v2 / cuCtxDestroy_v2  (driver context lifetime)
          activate/unload — load / unload model weights on GPU
        """
        if self.current_ctx_pct == mps:
            return

        # ── Step 1: create destination context if not yet existing
        self._ctx_create(mps)  # no-op if already exists

        # ── Step 2: schedule unload of old mps context (is None no action), then load new model.
        # _unload is non-blocking: starts grace-period timer so that quick switch back to `old` cancels unload with no H2D reload cost.
        self._unload(self.current_ctx_pct)
        self.current_ctx_pct = mps
        self._load(mps)

        # ── Step 3: background — pre-create neighbors, destroy far contexts ──
        def _background(current=mps):
            if current == 100:
                return
            keep = {current, 100} | {current + d for d in (-self.MPS_step, self.MPS_step) if 10 <= current + d <= 100}
            for n in keep - {current}:
                self._ctx_create(n)  # no-op if already exists
            with self._mps_ctx_lock: # Destroy contexts outside the keep window
                to_destroy = [p for p in list(self.ctx_states) if p not in keep]
            for p in to_destroy:
                with self._mps_ctx_lock:
                    state = self.ctx_states.get(p)
                    evt = self.ctx_unloading_events.get(p)
                    timer = self.ctx_deact_timers.get(p)
                self._destroy_ctx(p)
        threading.Thread(target=_background, daemon=True, name="ctx-bg").start()

    """
        Wait for scheduled executions from the client, stop until None is received.
        This should not be used by the llm_worker.
    """
    def sched_serve(self):
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time
        while(True):
            scheduling_plan = self.pipe.recv()
            sync = self.pipe.recv()
            if(scheduling_plan):
                scheduling_plan.execute(self)
            if(sync == SyncType.ALL):
                self.worker_barrier.wait()
            if(not scheduling_plan):
                break

    def run(self) -> None:
        self.init_start()
        self.submodules_setup()
        self.worker_barrier.wait()
        while(True):
            action = self.pipe.recv()
            sync = self.pipe.recv()
            stop_flag = False
            match action:
                case "profile":
                    self.profile()
                case "setup_testenv":
                    self.setup_testenv(self.pipe.recv())
                case "warmup":
                    self.test_config.set_warmup()
                    self.init_testenv_data()
                    self.warmup()
                case "test":
                    self.test_config.set_test()
                    self.test()
                case "sched_serve":
                    self.setup_testenv(self.pipe.recv())
                    self.init_testenv_data()
                    self.sched_serve()
                case "context_setup":
                    self.context_setup()
                case "get_total_modules":
                    self.pipe.send(self.total_module)
                case "collect_performance_data":
                    self.send_performance_data()
                case "collect_profiled_data":
                    self.send_profiled_data()
                case "stop":
                    stop_flag = True
                case _:
                    raise NotImplementedError
            if(sync == SyncType.SINGLE):
                self.single_worker_barrier.wait()
            elif(sync == SyncType.ALL):
                self.worker_barrier.wait()
            if(stop_flag):
                break
        self.shutdown()