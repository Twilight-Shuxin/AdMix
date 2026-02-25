import os
import torch
from torch.multiprocessing import Process
import time
import inspect
import numpy as np
from profiling.constants import *
from trace.trace_utils import SimpleTimeTrace
from profiling.models.workers.constants import SyncType

class ModelWorker(Process):
    def __init__(self, pipe, worker_barrier, single_worker_barrier):
        self.pipe = pipe
        self.worker_barrier = worker_barrier
        self.single_worker_barrier = single_worker_barrier
        self.device = "cuda"
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
    """
    def profile(self):
        profile_iter = 15
        warmup_iter = 5
        profile_data = {}
        for batch_size in self.supported_batchsize:
            spent_time = []
            for _ in range(warmup_iter):
                self.inference(self.random_input[batch_size], sync = True)
            for _ in range(profile_iter):
                start_time = time.perf_counter()
                self.inference(self.random_input[batch_size], sync = True)
                spent_time.append(time.perf_counter() - start_time)
            profile_data[batch_size] = np.percentile(spent_time, 99)
        self.pipe.send(profile_data)
    
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
        if(self.test_config.record_statistics):
            self.profile_data.logs = []
            self.profile_data.absolute_throughput_records = TimeBinRecord(init = 0, time_limit = self.test_config.test_time)
    
    def init_runtime_data(self):
        self.runtime_data = RuntimeData()
        self.runtime_data.absolute_throughput = 0
    
    def init_scheduler_data(self):
        self.scheduler_data = SchedulerData()
        self.scheduler_data.scheduler = get_scheduler(self.test_config.scheduler)

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
            if(self.test_config.record_statistics):
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
    
    """
        Wait for scheduled executions from the client, stop until None is received. 
        This should not be used by the llm_worker.
    """
    def sched_serve(self):
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time
        while(True):
            scheduling_plan = self.pipe.recv()
            if(not scheduling_plan):
                break
            scheduling_plan.execute(self)

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