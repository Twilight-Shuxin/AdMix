from profiling.models.model_worker import ModelWorker
from dataset.sharegpt_dataset import ShareGPTDataset
from trace.trace_utils import Trace, get_LLM_trace, get_trace_tag
from plotting.constants import *
import requests
import subprocess
import torch
import time
import signal
from profiling.constants import *
import os
import asyncio
import aiohttp
import json
import struct
import logging
import heapq
from plotting.trace_plot import *
from profiling.scheduling.monitor import *
from profiling.scheduling.utils import get_scheduler

def get_input_config_from_test_config(test_config):
    return InputConfig(seq_len=self.test_config.seq_len[0])

class LLMWorker(ModelWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("llm", device_id, mps, pipe, barrier, log_path, profile_path)
        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True
        }
        self.first_token_slo = 1
        self.idle_time_slo = 0

        self.SPEED_CONST = 0.2
        self.initial_time = self.SPEED_CONST

        self.server_proc = None
        self.intensity_server = None
        self.intensity_profiler = None
        self.scheduler_monitor = None
    
    def statistics_recording_init(self):
        self.profile_data.first_token_time = TimeBinRecord(init = [], time_limit = self.test_config.test_time)
        self.profile_data.between_token_time =  TimeBinRecord(init = [], time_limit = self.test_config.test_time)
        self.profile_data.failed = TimeBinRecord(init = 0, time_limit = self.test_config.test_time)
        self.profile_data.logs = []
    
    def init_start(self):
        # self.start_server()
        self.loop = asyncio.get_event_loop()
        self.dataset = ShareGPTDataset("shareGPT.json")

    def shutdown(self):
        # self.stop_server()
        return
    
    def start_server(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
        memory_util = "0.7"
        foo = os.environ.get('GVLLM_KV_MEM')
        if(foo is not None):
            memory_util = foo
        self.server_proc = subprocess.Popen(["python", "-m", "vllm.entrypoints.api_server"
            , "--model", "meta-llama/Llama-2-7b-chat-hf", "--gpu-memory-utilization", memory_util])
        session = requests.Session()
        session.trust_env = False
        ready_flag = False
        print("Starting server...")
        while(not ready_flag):
            try:
                response = session.get('http://localhost:8000/health') 
                if(response.status_code == 200):
                    ready_flag = True
            except:
                time.sleep(1)
        print("Server ready!")
    
    def stop_server(self):
        if(self.server_proc != None):
            self.server_proc.send_signal(signal.SIGINT)

    def get_input(self, input_config):
        prompt, prompt_len = self.dataset.get_query(input_config)
        return prompt_len, prompt
    
    def get_trace(self):
        if(not self.test_config.trace_path):
            return
    
        trace_config_id = int(self.test_config.trace_path)
        set_trace_config(trace_config_id)
        trace_config = read_LLM_trace_config()
        
        # Prioritize test time in cmd args, and sync test_config.test_time with trace_config.test_time
        if(self.test_config.test_time):
            trace_config.test_time = self.test_config.test_time
        elif(hasattr(trace_config, 'test_time')):
            self.test_config.test_time = trace_config.test_time
        else:
            trace_config["time"] = ModelWorker.DEFAULT_TEST_TIME
            self.test_config.test_time = ModelWorker.DEFAULT_TEST_TIME

        self.trace = LLMTrace(trace_config)
    
     """
        Get the tag of oneself in the format of:
            model_name(10,10)
            model_name(10,10_m70)
            model_name(m30)[trace_tag]
            model_name[trace_tag]
    """
    def get_self_tag(self, test_config):
        tag = self.model_name
        if(test_config.seq_len != None):
            subtag = ",".join(str(x) for x in test_config.seq_len)
        else:
            subtag = ""
        if(self.mps != None and self.mps != 100):
            if(len(subtag) != 0):
                subtag += "_"
            subtag += "m" + str(self.mps)
        if(len(subtag) != 0):
            tag = tag + "(" + subtag + ")"
        if(test_config.trace_path != None):
            tag = tag + "[" + get_trace_tag() + "]"
        return tag
    
    def warmup(self):
        warmup_config = TestConfig(None, 1, 0, self.total_module, None, [5,5])
        warmup_config.fit(self)
        self.random_test_period(warmup_config, 5)
    
    """
        Write basic logs:
            profile_data.logs: [(timestamp, prompt_tkn, output_tkn, ttft, latency, idle time, success)]
    """
    def write_basic_logs(self):
        self.profile_data.logs = sorted(self.profile_data.logs, key = lambda x: x[0])
        self.write_analytics("ftsummary.log", [x[3] for x in self.profile_data.logs])
        self.write_analytics("btsummary.log", [(x[4] / x[2]) for x in self.profile_data.logs])
        self.write_list_to_csv("all.csv", 
            ['Request timestamp (s)', 'Input token cnt', 'Output token cnt', 'TFT (s)', 'Latency (s)', 'Idle time (s)', 'Success'], 
            self.profile_data.logs)
    
    def write_logs(self):
        self.write_basic_logs()
        count_list_ftt, avg_list_ftt, max_list_ftt = self.profile_data.time_bins.first_token_time.collapse()
        count_list_btt, avg_list_btt, max_list_btt = self.profile_data.time_bins.between_token_time.collapse()
        tt_logs = [
            [id * self.profile_data.time_bins.first_token_time.bin_size, *values]
            for id, values in enumerate(zip(
                count_list_ftt, count_list_btt, avg_list_ftt,
                max_list_ftt, avg_list_btt, max_list_btt
            ))
        ]
        self.write_list_to_csv("ttsummary.csv", 
            ['Bin Timestamp (s)', 'FT throughput', 'BT throughput', 'Average TFT (s)', 'Max TFT (s)', 'Average TBT (s)', 'Max TBT(s)'], 
            tt_logs)
        
        avg_list_ftt = self.profile_data.time_bins.first_token_time.append_timestamps(avg_list_ftt)
        avg_list_btt = self.profile_data.time_bins.between_token_time.append_timestamps(avg_list_btt)
        failed_record_list = self.profile_data.time_bins.failed.append_timestamps()
        submitted_intensity = self.profile_data.submitted_record_counter.append_timestamps()
        waiting_first_intensity = self.profile_data.waiting_first_record_counter.append_timestamps()

        if(self.test_config.trace_path):
            self.write_all("submitted_intensity.log", submitted_intensity)
            self.write_all("failed_records.log", failed_record_list)
            trace_plot_two_intensity_with_analysis_data(submitted_intensity, "Submitted", waiting_first_intensity, "Waiting First",
                avg_list_btt, "Average BTT", ('Latency', '(s)'), self.get_file_path('intensity_bt_latency.png'))
            trace_plot_two_intensity_with_analysis_data(submitted_intensity, "Submitted", waiting_first_intensity, "Waiting First",
                avg_list_ftt, "Average FTT", ('Latency', '(s)'), self.get_file_path('intensity_ft_latency.png'))
            trace_plot_two_intensity_with_analysis_data_bar_plot(submitted_intensity, "Submitted", waiting_first_intensity, "Waiting First",
                failed_record_list, "Failed", ('Case Count', ''), self.get_file_path('submitted_failed.png'))
            trace_plot_aggregated_intensity(submitted_intensity, self.get_file_path('submitted.png'))
            trace_plot_aggregated_intensity_with_analysis_data(waiting_first_intensity, avg_list_btt,
                ('Latency', '(s)'), self.get_file_path('prefill_bt_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(submitted_intensity, avg_list_btt,
                ('Latency', '(s)'), self.get_file_path('submitted_bt_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(submitted_intensity, avg_list_ftt,
                ('Latency', '(s)'), self.get_file_path('submitted_ft_latency.png'))
    
    def check_performace(self, test_config):
        return self.total_submitted * 0.01 >= self.failed, self.completed, self.total_submitted - self.failed, self.total_submitted, (self.first_token_slo, self.idle_time_slo, self.SPEED_CONST, self.initial_time)

    # Return True: meet slo; False: violated slo
    def check_single_query_performance(self, request_timestamp, first_chunk_time, user_idle_time_info_record):
        def add_failed(timestamp):
            self.runtime_data.failed += 1
            if(self.test_config.record_statistics):
                self.profile_data.failed.add(timestamp, 1)
            return False

        if(request_timestamp == None):
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
                if((not user_idle_time_info_record) and (idle_time > self.idle_slo)):
                    user_idle_time_info_record = current_timestamp - user_idle_time + self.idle_slo
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
            last_refreshed_counter = self.scheduler_data.refreshed_flag

        try:
            async for chunk in receiver:
                current_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
                if(current_timestamp > self.test_config.test_time):
                    raise asyncio.CancelledError
                
                between_token_time = current_timestamp - last_received_time
                token_cnt += 1
                if(self.scheduler_data.scheduling_data_format == AIMDSchedulingDataFormat):
                    self.scheduler_data.uncompleted_tokens -= 1

                if not first_chunk_time:
                    first_chunk_time = between_token_time
                    if(self.test_config.record_statistics): 
                        self.profile_data.first_token_time.append_to_bin(current_timestamp, first_chunk_time)
                    self.runtime_data.waiting_first_record_counter.rm(current_timestamp)
                    
                    if(self.scheduler_data.scheduling_data_format == PredictiveSchedulingDataFormat):
                        self.scheduler_data.prefill_throughput_period += 1
                        self.scheduler_data.total_prefill_length -= input_length
                        self.scheduler_data.prefill_length_heap.delete((input_length, request_timestamp))
                        self.scheduler_data.total_decode_length += token_cnt
                else:
                    waiting_time += between_token_time
                    if(self.test_config.record_statistics):
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
                    self.scheduler_data.finished_tokens_period += received_token_tmp
                    self.scheduler_data.finished_count_period += 1
            if(self.scheduler_data.scheduling_data_format == AIMDScheduler):
                if(output_length != token_cnt - input_length):
                    self.scheduler_data.uncompleted_tokens -= output_length + input_length - token_cnt
            return first_chunk_time, user_idle_time, user_idle_time_info_record, 1
        except asyncio.CancelledError:
            current_timestamp = time.perf_counter() - self.runtime_data.start_time_pointer
            between_token_time = current_timestamp - last_received_time
            if(first_chunk_time):
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
      
    def profile(self):
        self.start_time_pointer = time.perf_counter()
        test_config = TestConfig(None, 1, 0, self.total_module, None, None)
        test_list = [[2000, 100], [5, 2000], [2000,2000]]
        for _ in range(1):
            for input_seqlen, output_seqlen in test_list:
                test_config.seq_len = [input_seqlen, output_seqlen]
                input = self.get_input(test_config)
                bt_rect = self.loop.run_until_complete(self.simple_inference(input, output_seqlen))   
                print(input_seqlen, output_seqlen, bt_rect)
        test_config = TestConfig(None, 1, 0, self.total_module, None, None)
        test_config.tag = "llm_PROFILE"
    
    async def unix_handler(self, reader, writer):
        _ = await reader.read(32)
        data = self.runtime_data.scheduling_data_format.pack_data(self)
        writer.write(len(data).to_bytes(4, "big"))
        await writer.drain()
        writer.write(data)
        await writer.drain()
        writer.close()
    
    async def intensity_server_unix(self):
        print("Starting Unix server")
        self.intensity_server = None
        self.intensity_server = await asyncio.start_unix_server(self.unix_handler, LLM_SOCKET_PATH)
    
    def init_scheduler_data(self):
        self.scheduler_data = SchedulerData()
        if(not self.test_config.scheduler):
            self.scheduler_data.scheduler_type = None
            return
        self.scheduler_data.scheduler_type = get_scheduler_class(self.test_config.scheduler)
        self.scheduler_data.scheduling_data_format = self.scheduler_data.scheduler_type.data_format

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
    
    def submodules_setup(self):
        self.total_module = -1
    
    def init_profile_data(self):
        self.profile_data = ProfileData()
        self.profile_data.completed = 0
        self.profile_data.failed = 0
        self.profile_data.total_submitted = 0
        self.profile_data.submitted_record_counter = RecordCounter(record_bins = self.test_config.record_statistics, 
                time_limit = self.test_config.test_time)
        self.profile_data.waiting_first_record_counter = RecordCounter(record_bins = self.test_config.record_statistics, 
                time_limit = self.test_config.test_time)
        
        self.queue_total_submitted_record = []
        self.queue_total_waiting_first_record = []

    def setup_testenv(self, test_config):
        self.test_config = test_config
        if(self.test_config.seq_len == None):
            self.test_config.seq_len = [10,10]
        self.get_trace()
        if(self.test_config.test_time is None):
            self.test_config.test_time = ModelWorker.DEFAULT_TEST_TIME
        self.input_config = get_input_config_from_test_config(self.test_config)
        self.test_config.full_tag = self.get_full_tag()
        self.runtime_data = RuntimeData()
        self.init_profile_data()
    
    async def _test(self):
        self.init_scheduler_data()
        if(self.test_config.scheduler):
            if(self.test_config.global_cnt > 1) # Serving mode
                asyncio.create_task(self.intensity_server_unix())
                if(self.scheduler_data.scheduler_type.monitor):
                    self.scheduler_data.scheduler_monitor = asyncio.create_task(self.scheduler_data.scheduler_type.monitor(self))
            else: # Profiling mode
                print("Scheduler profiling...")
                self.scheduler_data.intensity_profiler = asyncio.create_task(self.scheduler_data.scheduler_type.profiling_executor(self))
                
        task_list = []
        if(self.test_config.scheduler != None and self.test_config.global_cnt > 1):
            self.double_barrier.wait()
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time

        while(time.perf_counter() < self.runtime_data.time_limit):
            if(self.test_config.trace_path):
                req_time, sample_in_length, sample_out_length = self.trace.next()

                if(req_time < self.test_config.test_time):
                    sleep_time = req_time + self.runtime_data.start_time_pointer - time.perf_counter()
                    if(sleep_time > 0):
                        await asyncio.sleep(sleep_time)
                else: 
                    break
                input = self.get_input(InputConfig(seq_len=sample_in_length))
                task_list.append(asyncio.create_task(self.async_inference(input, sample_out_length)))
            else:
                input = self.get_input(InputConfig(seq_len=self.test_config.seq_len[0]))
                await self.async_inference(input, self.test_config.seq_len[1])

        if(self.runtime_data.intensity_server):
            self.runtime_data.intensity_server.close()
        if(self.runtime_data.intensity_profiler):
            self.runtime_data.intensity_profiler.cancel()
        if(self.runtime_data.scheduler_monitor):
            self.runtime_data.scheduler_monitor.cancel()
        
        for id, task in enumerate(task_list):
            if(not task.done()):
                task.cancel()
    
    def test(self):
        if(self.test_config.record_statistics):
            self.statistics_recording_init()
        if(self.test_config.record_dcgm):
            dcgm_monitor_conn = self.start_dcgm()
        
        self.loop.run_until_complete(self._test())

        if(self.test_config.record_dcgm):
            stop_dcgm(dcgm_monitor_conn)

        if(self.test_config.record_statistics):
            self.write_logs()
    

