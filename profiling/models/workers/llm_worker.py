import os
import time
import torch
import signal
import asyncio
import aiohttp
import requests
import subprocess
from profiling.constants import *
from profiling.models.workers.model_worker import ModelWorker
from dataset.sharegpt_dataset import ShareGPTDataset
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config

class LLMWorker(ModelWorker):
    def __init__(self, pipe, worker_barrier, single_worker_barrier, slo_info):
        super().__init__(pipe, worker_barrier, single_worker_barrier)
        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True
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
    
    def start_server(self):
        os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
        memory_util = os.environ.get('GVLLM_KV_MEM', "0.7")
        use_modified = os.environ.get('GVLLM_USE_MODIFIED', 'False').lower() == 'true'
        system_setup_cmd = "bash vllm_setup.sh"
        if(use_modified):
            system_setup_cmd += " modified"
        os.system(system_setup_cmd)
        self.server_proc = subprocess.Popen(["python", "-m", "vllm.entrypoints.api_server", 
            "--model", "meta-llama/Llama-2-7b-chat-hf", "--tensor-parallel-size", str(torch.cuda.device_count()),
            "--gpu-memory-utilization", memory_util, "--no-enable-prefix-caching"])

        session = requests.Session()
        session.trust_env = False
        print("Starting server...")
        while(True):
            try:
                response = session.get('http://localhost:8000/health') 
                if(response.status_code == 200):
                    break
            except:
                time.sleep(1)
        print("Server ready!")
    
    def stop_server(self):
        if(self.server_proc):
            self.server_proc.send_signal(signal.SIGINT)
    
    def init_profile_data(self):
        self.profile_data = ProfileData()
        if(self.test_config.record_statistics):
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
        if(self.test_config.record_statistics):
            self.runtime_data.submitted_record_counter.set_record_bins(TimeBinRecord(init = [], time_limit = self.test_config.test_time))
            self.runtime_data.waiting_first_record_counter.set_record_bins(TimeBinRecord(init = [], time_limit = self.test_config.test_time))
    
    @classmethod
    def get_trace_config(cls, test_config):
        if(not test_config.trace_path):
            return None
        trace_config_id = int(test_config.trace_path)
        set_LLM_trace_config(trace_config_id)
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
            if(self.test_config.record_statistics):
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

        try:
            async for chunk in receiver:
                current_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
                if(current_timestamp > self.test_config.test_time):
                    raise asyncio.CancelledError
                
                between_token_time = current_timestamp - last_received_time
                token_cnt += 1

                if first_chunk_time is None:
                    first_chunk_time = between_token_time
                    if(self.test_config.record_statistics): 
                        self.profile_data.first_token_time.append_to_bin(current_timestamp, first_chunk_time)
                    self.runtime_data.waiting_first_record_counter.rm(current_timestamp)
                else:
                    waiting_time += between_token_time
                    if(self.test_config.record_statistics):
                        self.profile_data.between_token_time.append_to_bin(current_timestamp, between_token_time)
                    
                    user_idle_time, user_idle_time_info_record = update_user_idle_time_info_record(
                        user_idle_time, user_idle_time_info_record, current_timestamp, waiting_time - reading_time)
                    reading_time += self.SPEED_CONST

                last_received_time = current_timestamp
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
        completed = False

        try:
            request_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
            current_timestamp = request_timestamp
            self.runtime_data.submitted_record_counter.add(current_timestamp)
            self.runtime_data.waiting_first_record_counter.add(current_timestamp)

            timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                generation_input = {
                    "prompt": prompt,
                    "stream": self.server_config['stream'],
                    "ignore_eos": True,
                    "max_tokens": max_tokens,
                    "temperature": self.server_config['temperature'],
                }
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
            if(self.test_config.record_statistics):
                self.profile_data.logs.append([request_timestamp, prompt_len, max_tokens, first_chunk_time, latency, user_idle_time, result])
        except asyncio.CancelledError:
            raise
    
    async def _test(self):
        self.init_testenv_data()

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
        
        for id, task in enumerate(task_list):
            if(not task.done()):
                task.cancel()

        await asyncio.gather(*task_list, return_exceptions=True)

    