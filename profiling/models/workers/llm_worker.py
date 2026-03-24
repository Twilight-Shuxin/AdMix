import os
import time
import torch
import signal
import asyncio
import aiohttp
import requests
import subprocess
from profiling.constants import *
from dataset.sharegpt_dataset import ShareGPTDataset
from profiling.models.workers.model_worker import ModelWorker
from profiling.scheduling.utils import get_scheduler_class
from profiling.scheduling.monitor import LLM_SERVER_SOCKET_PATH, LLM_INTENSITY_SOCKET_PATH
from profiling.scheduling import AIMDScheduler, AIMDSchedulingDataFormat, PredictiveScheduler, PredictiveSchedulingDataFormat, BatchScheduler
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config

modify_cnt = 0 
#FIXME:: Only used for testing
def modify_prompt(prompt):
    return prompt
    global modify_cnt
    if len(prompt) > 2 and modify_cnt % 2 == 0:
        prompt = prompt[1:]
    if len(prompt) > 2 and modify_cnt % 3 == 0:
        prompt = prompt[1:]
    modify_list = ["The", "Hi", "Okay", "你好", "不啊", "私", "今日", "How", "All", "May", "Why", "Now", "Go", "早", "***", "ke", "1o1k1", "uu", "ytq", "????", "$$"]
    prompt = modify_list[modify_cnt % len(modify_list)] + prompt
    modify_cnt = (modify_cnt + 1) % len(modify_list)
    return prompt


class LLMWorker(ModelWorker):
    def __init__(self, devices, pipe, worker_barrier, single_worker_barrier, log_path, profile_path, slo_info):
        super().__init__(devices, pipe, worker_barrier, single_worker_barrier, log_path, profile_path)
        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True,
            'server_type': os.environ.get('ADMIX_LLM_SERVING_PLATFORM', "vllm").lower()
        }
        self.server_proc = None
        self.first_token_slo, self.idle_time_slo, self.SPEED_CONST = slo_info
        self.initial_time = self.SPEED_CONST
    
    def init_start(self):
        self.start_server()
        self.loop = asyncio.get_event_loop()
        self.dataset = ShareGPTDataset("shareGPT.json")
    
    def shutdown(self):
        self.stop_server()
        return
    
    def _wait_for_server(self, health_url):
        session = requests.Session()
        session.trust_env = False
        print("Starting server...")
        while True:
            try:
                if session.get(health_url).status_code == 200:
                    break
            except:
                time.sleep(1)
        print("Server ready!")

    def start_vllm_server(self):
        os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
        memory_util = os.environ.get('ADMIX_LLM_KV_MEM', "0.7")
        self.server_proc = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", "meta-llama/Llama-2-7b-chat-hf",
            "--tensor-parallel-size", str(torch.cuda.device_count()),
            "--gpu-memory-utilization", memory_util,
            "--no-enable-prefix-caching",
        ])
        self._wait_for_server(f"http://localhost:{self.server_config['port']}/health")

    def start_sglang_server(self):
        memory_util = os.environ.get('ADMIX_LLM_KV_MEM', "0.7")
        env = os.environ.copy()
        env.update({"NO_PROXY": "0.0.0.0,localhost,127.0.0.1", "no_proxy": "0.0.0.0,localhost,127.0.0.1"})
        self.server_proc = subprocess.Popen([
            "uv", "run", "--project", "plugins/sglang_env",
            "python", "-m", "sglang.launch_server",
            "--model-path", "meta-llama/Llama-2-7b-chat-hf",
            "--chunked-prefill-size", "2048",
            "--enable-mixed-chunk",
            "--tp", str(torch.cuda.device_count()),
            "--mem-fraction-static", memory_util,
            "--port", str(self.server_config['port']),
            "--disable-radix-cache",
            "--log-level", "warning",
            "--disable-overlap-schedule"
        ], env=env, start_new_session=True)
        self._wait_for_server(f"http://localhost:{self.server_config['port']}/health")

    def start_server(self):
        use_modified = os.environ.get('ADMIX_LLM_MODIFIED', 'False').lower() == 'true'
        system_setup_cmd = "bash llm_platform_setup.sh"
        if use_modified:
            system_setup_cmd += " modified"
        os.system(system_setup_cmd)
        match self.server_config['server_type']:
            case "vllm":
                self.start_vllm_server()
            case "sglang":
                self.start_sglang_server()

    def build_generation_input(self, prompt, max_tokens):
        input_message = None
        match self.server_config['server_type']:
            case 'sglang':
                input_message = {
                    "text": prompt,
                    "stream": self.server_config['stream'],
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "temperature": self.server_config['temperature'],
                        "ignore_eos": True,
                    },
                }
            case 'vllm':
                input_message = {
                    "prompt": prompt,
                    "stream": self.server_config['stream'],
                    "ignore_eos": True,
                    "max_tokens": max_tokens,
                    "temperature": self.server_config['temperature'],
                }
        return input_message
    
    def stop_server(self):
        if(self.server_proc):
            os.killpg(os.getpgid(self.server_proc.pid), signal.SIGINT)
            self.server_proc.wait()
    
    def init_profile_data(self):
        self.profile_data = ProfileData()
        if(self.test_config.recording_action.record_statistics):
            self.profile_data.failed_records = TimeBinRecord(init = 0, time_limit = self.test_config.test_time)
            self.profile_data.first_token_time = TimeBinRecord(init = [], time_limit = self.test_config.test_time)
            self.profile_data.between_token_time =  TimeBinRecord(init = [], time_limit = self.test_config.test_time)
            self.profile_data.logs = []
    
    def init_runtime_data(self):
        self.runtime_data = RuntimeData()
        self.runtime_data.completed = 0
        self.runtime_data.total_submitted = 0
        self.runtime_data.failed = 0
        self.runtime_data.submitted_record_counter = RecordCounter()
        self.runtime_data.waiting_first_record_counter = RecordCounter()
        if(self.test_config.recording_action.record_statistics):
            self.runtime_data.submitted_record_counter.set_record_bins(TimeBinRecord(init = [], time_limit = self.test_config.test_time))
            self.runtime_data.waiting_first_record_counter.set_record_bins(TimeBinRecord(init = [], time_limit = self.test_config.test_time))
    
    def init_scheduler_data(self):
        self.scheduler_data = SchedulerData()
        self.scheduler_data.scheduler_type = None
        self.scheduler_data.scheduling_data_format = None
        self.scheduler_data.intensity_server = None
        self.scheduler_data.scheduler_profiler = None
        self.scheduler_data.scheduler_monitor = None

        if(not self.test_config.scheduler):
            return
        self.scheduler_data.scheduler_type = get_scheduler_class(self.test_config.scheduler)
        self.scheduler_data.scheduling_data_format = self.scheduler_data.scheduler_type.data_format
        self.scheduler_data.opp_ready = asyncio.Event()

        # basic_aimd
        if(self.scheduler_data.scheduler_type == AIMDScheduler):
            self.scheduler_data.uncompleted_tokens = 0

        # predict
        if(self.scheduler_data.scheduler_type == PredictiveScheduler):
            # point data
            self.scheduler_data.total_prefill_length = 0
            self.scheduler_data.prefill_length_heap = DeletableMaxHeap()
            self.scheduler_data.total_decode_length = 0

            # period data
            self.scheduler_data.prefill_throughput_period = 0
            self.scheduler_data.decode_throughput_period = 0
            self.scheduler_data.prefill_length_period = 0
            self.scheduler_data.prefill_length_heap_period = MaxHeap()
            self.scheduler_data.finished_count_period = 0
            self.scheduler_data.finished_tokens_period = 0
            self.scheduler_data.refreshed_counter = 0
    
    @classmethod
    def get_trace_config(cls, test_config):
        if(not test_config.trace_path):
            return None
        set_LLM_trace_config(test_config.trace_path)
        trace_config = read_LLM_trace_config()
        # Prioritize test time in cmd args, and sync test_config.test_time with trace_config.test_time
        if(not (test_config.test_time_default and hasattr(trace_config, 'test_time'))):
            trace_config.test_time = test_config.test_time
        return trace_config
    
    def get_trace(self):
        if(not self.test_config.trace_path):
            return
        trace_config = self.get_trace_config(self.test_config)
        self.test_config.test_time = trace_config.test_time
        self.test_config.trace = LLMTrace(trace_config)

    @classmethod
    def get_input_config_from_test_config(cls, test_config):
        return InputConfig(seq_len = test_config.seq_len[0] if test_config.seq_len else None)

    def get_input(self, input_config):
        prompt, prompt_len = self.dataset.get_query(input_config)
        return prompt, prompt_len
    
    """
        Perform inferences for GPU & model warmup.
    """
    def warmup(self):
        return # FIXME:: Currently ignored for debugging
        warmup_iter = 10
        llm_input = self.get_input(InputConfig(seq_len = 10))
        self.runtime_data.start_time_pointer = time.perf_counter()
        for _ in range(warmup_iter):
            self.loop.run_until_complete(self.async_inference(llm_input, 10))
    
    def send_performance_data(self):
        self.pipe.send(self.runtime_data)
    
    def send_profiled_data(self):
        self.pipe.send(self.profile_data)
    
    # Return True: meet slo; False: violated slo
    def check_single_query_performance(self, request_timestamp, first_chunk_time, user_idle_time_info_record):
        def add_failed(timestamp):
            self.runtime_data.failed += 1
            if(self.test_config.recording_action.record_statistics):
                self.profile_data.failed_records.add(timestamp, 1)
            return False

        if(request_timestamp is None):
            return True
        self.runtime_data.total_submitted += 1

        if(first_chunk_time > self.first_token_slo):
            return add_failed(request_timestamp + self.first_token_slo)
        elif(user_idle_time_info_record):
            return add_failed(user_idle_time_info_record)
        return True

    async def stream_response(self, receiver, request_timestamp, input_length, output_length):
        
        def update_user_idle_time_info_record(user_idle_time, user_idle_time_info_record, current_timestamp, idle_time):
            if(idle_time > user_idle_time):
                user_idle_time = idle_time
                if((not user_idle_time_info_record) and (idle_time > self.idle_time_slo)):
                    user_idle_time_info_record = current_timestamp - user_idle_time + self.idle_time_slo
            return user_idle_time, user_idle_time_info_record

        first_chunk_time = None
        user_idle_time = 0
        user_idle_time_info_record = None
        last_received_time = request_timestamp
        reading_time = self.initial_time
        waiting_time = 0
        
        token_cnt = input_length

        if(self.scheduler_data.scheduling_data_format == PredictiveSchedulingDataFormat):
            received_token_period = 0
            last_refreshed_counter = self.scheduler_data.refreshed_counter

        try:
            first = None
            async for chunk in receiver:
                if(not first):
                    first = chunk
                current_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
                if(current_timestamp > self.test_config.test_time):
                    raise asyncio.CancelledError
                
                between_token_time = current_timestamp - last_received_time
                token_cnt += 1
                if(self.scheduler_data.scheduling_data_format == AIMDSchedulingDataFormat):
                    self.scheduler_data.uncompleted_tokens -= 1

                if first_chunk_time is None:
                    first_chunk_time = between_token_time
                    if(self.test_config.recording_action.record_statistics): 
                        self.profile_data.first_token_time.append_to_bin(current_timestamp, first_chunk_time)
                    self.runtime_data.waiting_first_record_counter.rm(current_timestamp)
                    
                    if(self.scheduler_data.scheduling_data_format == PredictiveSchedulingDataFormat):
                        self.scheduler_data.prefill_throughput_period += 1
                        self.scheduler_data.total_prefill_length -= input_length
                        self.scheduler_data.prefill_length_heap.delete((input_length, request_timestamp))
                        self.scheduler_data.total_decode_length += token_cnt
                else:
                    waiting_time += between_token_time
                    if(self.test_config.recording_action.record_statistics):
                        self.profile_data.between_token_time.append_to_bin(current_timestamp, between_token_time)
                    
                    if(self.scheduler_data.scheduling_data_format == PredictiveSchedulingDataFormat):
                        if(last_refreshed_counter != self.scheduler_data.refreshed_counter):
                            received_token_period = 0
                            last_refreshed_counter = self.scheduler_data.refreshed_counter
                        received_token_period += 1
                        self.scheduler_data.total_decode_length += 1
                        self.scheduler_data.decode_throughput_period += 1
                    
                    user_idle_time, user_idle_time_info_record = update_user_idle_time_info_record(
                        user_idle_time, user_idle_time_info_record, current_timestamp, waiting_time - reading_time)
                    reading_time += self.SPEED_CONST

                last_received_time = current_timestamp
            if(self.scheduler_data.scheduling_data_format == PredictiveSchedulingDataFormat):
                self.scheduler_data.total_decode_length -= token_cnt
                if(last_refreshed_counter == self.scheduler_data.refreshed_counter):
                    self.scheduler_data.finished_tokens_period += received_token_period
                    self.scheduler_data.finished_count_period += 1
            if(self.scheduler_data.scheduling_data_format == AIMDSchedulingDataFormat):
                if(output_length != token_cnt - input_length):
                    self.scheduler_data.uncompleted_tokens -= output_length + input_length - token_cnt
            return first_chunk_time, user_idle_time, user_idle_time_info_record, 1
        except asyncio.CancelledError:
            current_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
            between_token_time = current_timestamp - last_received_time
            if(first_chunk_time is not None):
                waiting_time += between_token_time
                user_idle_time, user_idle_time_info_record = update_user_idle_time_info_record(
                    user_idle_time, user_idle_time_info_record, current_timestamp, waiting_time - reading_time)
            else:
                first_chunk_time = between_token_time
            return first_chunk_time, user_idle_time, user_idle_time_info_record, 0

    # input: (prompt, prompt len)
    async def async_inference(self, input, max_tokens):
        prompt, prompt_len = input
        first_chunk_time = None 
        user_idle_time = 0 
        user_idle_time_info_record = None
        request_timestamp = None
        if(self.scheduler_data.scheduler_type == AIMDScheduler):
            self.scheduler_data.uncompleted_tokens += max_tokens
        completed = False

        try:
            request_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
            current_timestamp = request_timestamp
            self.runtime_data.submitted_record_counter.add(current_timestamp)
            self.runtime_data.waiting_first_record_counter.add(current_timestamp)
                      
            if(self.scheduler_data.scheduling_data_format == PredictiveSchedulingDataFormat):
                self.scheduler_data.total_prefill_length += prompt_len
                self.scheduler_data.prefill_length_period += prompt_len
                self.scheduler_data.prefill_length_heap.push((prompt_len, request_timestamp))
                self.scheduler_data.prefill_length_heap_period.push((prompt_len, request_timestamp))

            timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                generation_input = self.build_generation_input(prompt, max_tokens)
                async with session.post(
                    f"http://{self.server_config['host']}:{self.server_config['port']}/generate", json=generation_input
                ) as resp:
                    if resp.status != 200:
                        print(f"Error: {resp.status} {resp.reason}")
                        print(await resp.text())
                        return
                    first_chunk_time, user_idle_time, user_idle_time_info_record, completed = await \
                        self.stream_response(resp.content.iter_any(), request_timestamp, prompt_len, max_tokens)
            
            current_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
            self.runtime_data.submitted_record_counter.rm(current_timestamp)
            if(completed):
                self.runtime_data.completed += 1
                latency = current_timestamp - request_timestamp
            else:
                latency = -1
            result = self.check_single_query_performance(request_timestamp, first_chunk_time, user_idle_time_info_record)
            if(self.test_config.recording_action.record_statistics):
                self.profile_data.logs.append([request_timestamp, prompt_len, max_tokens, first_chunk_time, latency, user_idle_time, result])
        except asyncio.CancelledError:
            raise
    
    async def create_unix_server(self):
        async def server_unix_handler(reader, writer):
            action = (await reader.read(32)).decode("utf-8")
            match action:
                case "test_time":
                    writer.write(self.test_config.test_time.to_bytes(4, "big"))
                    await writer.drain()
                case "ready":
                    self.scheduler_data.opp_ready.set()
                case "tag":
                    writer.write(self.test_config.self_full_tag.encode("utf-8"))
                    await writer.drain()
            writer.close()
            await writer.wait_closed()
    
        async def intensity_unix_handler(reader, writer):
            _ = await reader.read(32)
            data = self.scheduler_data.scheduling_data_format.pack_data(self)
            writer.write(len(data).to_bytes(4, "big"))
            await writer.drain()
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        print("Starting Unix servers")
        self.scheduler_data.general_server = await asyncio.start_unix_server(server_unix_handler, LLM_SERVER_SOCKET_PATH)
        if(self.scheduler_data.scheduler_type != BatchScheduler):
            self.scheduler_data.intensity_server = await asyncio.start_unix_server(intensity_unix_handler, LLM_INTENSITY_SOCKET_PATH)
    
    async def _test(self):
        self.init_testenv_data()
        await self.create_unix_server()

        if(self.test_config.scheduler):
            if(self.test_config.global_cnt > 1): # Serving mode
                if(self.scheduler_data.scheduler_type.monitor):
                    self.scheduler_data.scheduler_monitor = asyncio.create_task(self.scheduler_data.scheduler_type.monitor(self))
                # Be informed by opportunistic worker that the other side have retrieved necessary information and is ready to move on
                await self.scheduler_data.opp_ready.wait()
            else: # Profiling mode
                print("Scheduler profiling...")
                self.scheduler_data.scheduler_profiler = asyncio.create_task(self.scheduler_data.scheduler_type.profiling_executor(self))
                
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time
        task_list = []

        while(time.perf_counter() < self.runtime_data.time_limit):
            if(self.test_config.trace_path):
                req_time, sample_in_length, sample_out_length = self.test_config.trace.next()
                sleep_time = min(req_time, self.test_config.test_time) + self.runtime_data.start_time_pointer - time.perf_counter()
                if(sleep_time > 0):
                    await asyncio.sleep(sleep_time)
                if(req_time >= self.test_config.test_time):
                    break
                llm_input = self.get_input(InputConfig(seq_len = sample_in_length))
                task_list.append(asyncio.create_task(self.async_inference(llm_input, sample_out_length)))
            else:
                llm_input = self.get_input(self.input_config)
                await self.async_inference(llm_input, self.test_config.seq_len[1])
        
        self.scheduler_data.general_server.close()
        if(self.scheduler_data.intensity_server):
            self.scheduler_data.intensity_server.close()
        if(self.scheduler_data.scheduler_profiler):
            print("Close scheduler profiler")
            self.scheduler_data.scheduler_profiler.cancel()
        if(self.scheduler_data.scheduler_monitor):
            self.scheduler_data.scheduler_monitor.cancel()
        
        for id, task in enumerate(task_list):
            if(not task.done()):
                task.cancel()
