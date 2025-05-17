from profiling.models.model_worker import ModelWorker
from dataset.new_sharegpt_dataset import NewShareGPTDataset
from trace.trace_utils import BurstGPTTrace,Trace
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
import logging
from plotting.trace_plot import *

def get_bin_list(target_list):
    return [(i * BIN_SIZE, value) for i, value in enumerate(target_list)]

class LLMWorker(ModelWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path):
        super().__init__("llm", device_id, mps, pipe, barrier, log_path)
        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True
        }
        self.logs = []
        self.SPEED_CONST = 0.25
        self.server_proc = None
        self.time_intervals = []
        self.time_bins = None
        self.first_token_slo = 1
        self.idle_time_slo = 0.5
        self.total_submitted = 0
        self.failed = 0
        self.completed = 0
        # total_submitted & waiting_first
        self.queue_record = [0, 0]
        self.queue_total_submitted_record = []
        self.queue_total_waiting_first_record = []
    
    def recording_queue_records(self, target_id, target_list, timestamp, value):
        self.queue_record[target_id] += value
        target_list.append((timestamp, self.queue_record[target_id]))
    
    def init_start(self):
        # self.start_server()
        self.loop = asyncio.get_event_loop()
        self.dataset = NewShareGPTDataset("shareGPT.json")

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
        # self.server_proc = subprocess.Popen(["python", "-m" ,"vllm.entrypoints.api_server"
        #     ,"--model", "meta-llama/Llama-2-7b-chat-hf"])
        session = requests.Session()
        session.trust_env = False
        ready_flag = False
        print("starting server")
        while(ready_flag == False):
            try:
                response = session.get('http://localhost:8000/health') 
                if(response.status_code == 200):
                    ready_flag = True
            except:
                time.sleep(1)
        print("server ready")
    
    def stop_server(self):
        if(self.server_proc != None):
            self.server_proc.send_signal(signal.SIGINT)

    def get_input(self, test_config):
        return self.dataset.get_query(test_config)[0]
    
    def get_trace(self, trace_path):
        self.trace = BurstGPTTrace(trace_path)
    
    def get_tag(self, test_config):
        tag = self.model_name
        if(test_config.seq_len != -1):
            subtag = ",".join(str(x) for x in test_config.seq_len)
        else:
            subtag = "n"
        if(self.mps != -1 and self.mps != 100):
            subtag += "_m" + str(self.mps)
        if(len(subtag) != 0):
            tag = tag + "(" + subtag + ")"
        return tag
    
    def warmup(self):
        warmup_config = TestConfig(1, None, 1, 0, self.total_module, -1, [5,5])
        warmup_config.fit(self)
        self.random_test_period(warmup_config, 5)
    
    def write_log(self, test_config):
        self.write_analytics(test_config, "ftsummary.log", [x[1] for x in self.logs])
        self.write_analytics(test_config, "atsummary.log", [x[2] for x in self.logs])
        self.write_analytics(test_config,"idlesummary.log", [x[3] for x in self.logs])
        self.logs = sorted(self.logs, key = lambda x: x[0])

        log_filepath = self.get_file_path(test_config, "all.log")  
        with open(log_filepath, "w+") as log_output:
            for log_record in self.logs: 
                log_output.write(list_to_string(log_record) + "\n")
        
        log_filepath = self.get_file_path(test_config,  "ttsummary.log")  
        average_ftt_list = []
        average_btt_list = []
        throughput_list = []
        with open(log_filepath, "w+") as log_output:
            for id, bin in enumerate(self.time_bins): 
                bin_start = id * BIN_SIZE
                average_ftt = np.average(bin[0]) if len(bin[0]) > 0 else 0
                average_btt = np.average(bin[1]) if len(bin[1]) > 0 else 0
                average_ftt_list.append(average_ftt)
                average_btt_list.append(average_btt)
                throughput_list.append(len(bin[1]))
                log_output.write(str(bin_start) + " " + str(len(bin)) + " " + str(average_ftt) + " " + str(average_btt) + "\n")
        
        average_ftt_list = get_bin_list(average_ftt_list)
        average_btt_list = get_bin_list(average_btt_list)
        throughput_list = get_bin_list(throughput_list)
        if(test_config.trace_path != None):
            trace_plot_intensity_with_analysis_data(self.queue_total_submitted_record, self.queue_total_waiting_first_record, 
                average_btt_list, self.get_file_path(test_config, 'intensity_bt_latency.png'))
            trace_plot_intensity_with_analysis_data(self.queue_total_submitted_record, self.queue_total_waiting_first_record, 
                average_ftt_list, self.get_file_path(test_config, 'intensity_ft_latency.png'))
            trace_plot_aggregated_intensity(self.time_intervals, self.get_file_path(test_config, 'aggregated.png'))
            trace_plot_aggregated_intensity_with_analysis_data(self.time_intervals, average_btt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'aggregated_bt_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(self.time_intervals, average_ftt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'aggregated_ft_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(self.time_intervals, throughput_list,
                ('Throughput', ''), self.get_file_path(test_config, 'aggregated_throughput.png'))
    
    def check_performace(self, test_config):
        log_filepath = os.path.join(self.log_path, "-".join([test_config.tag, "all.log"]))
        return self.total_submitted * 0.01 >= self.failed, self.completed, self.total_submitted - self.failed, self.total_submitted, (self.first_token_slo, self.idle_time_slo)

    def check_single_query_performance(self, request_timestamp, first_chunk_time, user_idle_time, start_time, last_received_time, reading_time):
        if(start_time == None):
            return
        failed_flag = False
        self.total_submitted += 1
        self.time_intervals.append((request_timestamp, time.perf_counter() - start_time + request_timestamp))
        if(first_chunk_time == None):
            first_chunk_time = time.perf_counter() - start_time
        if(last_received_time != None):
            user_idle_time = max(user_idle_time, time.perf_counter() - last_received_time - reading_time)
        failed_flag = (first_chunk_time > self.first_token_slo)
        failed_flag = failed_flag or (user_idle_time > self.idle_time_slo)
        if(failed_flag):
            self.failed += 1
    
    async def _inference(self, input, rec_config, max_tokens):
        user_idle_time = 0
        first_chunk_time = 0
        start_time = None
        reading_time = 0
        waiting_time = 0
        last_received_time = None
        try:
            # no start & end available for LLMworker served by vllm
            timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                generation_input = {
                    "prompt": input,
                    "stream": self.server_config['stream'],
                    "ignore_eos": True,
                    "max_tokens": max_tokens,
                    "temperature": self.server_config['temperature'],
                }
                if(rec_config[0] != False):
                    self.recording_queue_records(0, self.queue_total_submitted_record, rec_config[1], 1)
                    self.recording_queue_records(1, self.queue_total_waiting_first_record, rec_config[1], 1)
                reading_time = 0
                start_time = time.perf_counter()
                failed_flag = False
                async with session.post(
                    f"http://{self.server_config['host']}:{self.server_config['port']}/generate", json=generation_input
                ) as resp:
                    if resp.status != 200:
                        print(f"Error: {resp.status} {resp.reason}")
                        print(await resp.text())
                        return None, None, None

                    if self.server_config['stream']:
                        buffer = b""
                        first_chunk_received = False
                        async for chunk in resp.content.iter_any():
                            buffer = chunk
                            # If this is the first chunk, record the time taken
                            if(rec_config[0] != False):
                                current_time_stamp = rec_config[1] + time.perf_counter() - start_time
                            if not first_chunk_received:
                                first_chunk_time = time.perf_counter() - start_time
                                first_chunk_received = True
                                if(rec_config[0] != False): 
                                    self.time_bins[int(current_time_stamp / BIN_SIZE)][0].append(first_chunk_time)
                                    self.recording_queue_records(1, self.queue_total_waiting_first_record, current_time_stamp, -1)
                            else:
                                between_token_time = time.perf_counter() - last_received_time
                                waiting_time += between_token_time
                                if(rec_config[0] != False): 
                                    self.time_bins[int(current_time_stamp / BIN_SIZE)][1].append(between_token_time)
                                user_idle_time = max(user_idle_time, waiting_time - reading_time)
                                reading_time += self.SPEED_CONST
                            last_received_time = time.perf_counter()
                        # output = json.loads(buffer.decode("utf-8"))  # Decode JSON
                    else:
                        output = await resp.json()  
            if(rec_config[0] != False):
                self.completed += 1
                self.recording_queue_records(0, self.queue_total_submitted_record, current_time_stamp, -1)
                self.logs.append([rec_config[1], first_chunk_time, time.perf_counter() - start_time, user_idle_time])
                self.check_single_query_performance(rec_config[1], first_chunk_time, user_idle_time, start_time, last_received_time, reading_time)
        except asyncio.CancelledError:
            if(rec_config[0] != False):
                self.check_single_query_performance(rec_config[1], first_chunk_time, user_idle_time, start_time, last_received_time, reading_time)
            raise

    def inference(self, input, rec_config = (False,), max_tokens = 15):
        self.loop.run_until_complete(self._inference(input, rec_config, max_tokens))

    def random_test_period(self, test_config, time_len, rec = RecordingAction.NO_RECORD):
        start_time = time.perf_counter()
        if(test_config.seq_len == -1):
            test_config.seq_len = [10, 10]
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        while(time.perf_counter() - start_time < time_len):
            input = self.get_input(test_config)
            self.inference(input, (rec in RECORDING_ACTION_MAP["statistics"],))
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)

    async def trace_test_period(self, test_config, time_len, rec = RecordingAction.NO_RECORD):
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        if(rec in RECORDING_ACTION_MAP["statistics"] and self.time_bins == None):
            # first list for first_token_time and second list for time_between_token
            self.time_bins = [[[],[]] for x in np.arange(0, time_len + BIN_SIZE * 3, BIN_SIZE)]
        test_start_time = time.perf_counter()
        time_limit = test_start_time + time_len
        task_list = []
        while(True):
            req_time, sample_in_length, sample_out_length = self.trace.next()
            test_config.seq_len = [sample_in_length, sample_out_length]
            sleep_time = req_time + test_start_time - time.perf_counter()
            if(sleep_time > 0):
                await asyncio.sleep(sleep_time)
            if(time.perf_counter() > time_limit):
                break
            input = self.get_input(test_config)
            task_list.append(asyncio.create_task(self._inference(input, 
                (rec in RECORDING_ACTION_MAP["statistics"], req_time, time_limit), sample_out_length)))
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)
        # await asyncio.gather(*task_list)
        for id, task in enumerate(task_list):
            if(not task.done()):
                task.cancel()
        # print("processed:", cnt, "in", time.perf_counter() - test_start_time, "(s) uncompleted:", uncompleted)
    
    def common_start(self):
        self.total_module = -1
    
    def random_test(self, test_config):
        self.random_test_period(test_config, self.warmup_time)
        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        recording_action = RecordingAction.ALL if test_config.recording_action == None else test_config.recording_action
        if(test_config.trace_path != None):
            self.loop.run_until_complete(self.trace_test_period(test_config, test_time, recording_action))
        else:
            self.random_test_period(test_config, test_time, recording_action)
        self.random_test_period(test_config, self.warmup_time)
        if(recording_action in RECORDING_ACTION_MAP["statistics"]):
            self.write_log(test_config)
    

