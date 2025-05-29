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
import struct
import logging
from plotting.trace_plot import *
from profiling.scheduling.monitor import *
from profiling.scheduling.scheduler import schedulers

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
        self.record_throughput = 0
        self.throughput_counter = 0
        self.record_intensity = 0
        self.record_intensity_prev = 0
        # total_submitted & waiting_first
        self.queue_record = [0, 0]
        self.queue_total_submitted_record = []
        self.queue_total_waiting_first_record = []
        
    def recording_queue_records(self, rec, target_id, target_list, timestamp, value):
        self.queue_record[target_id] += value
        if(rec != False):
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
        self.write_all(test_config, "all.log", self.logs)
        
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
            aggregated_intensity = get_bin_from_intervals(self.time_intervals)
            self.write_all(test_config, "aggregated_intensity_all.log", list(map(list, zip(*aggregated_intensity))))
            trace_plot_intensity_with_analysis_data(self.queue_total_submitted_record, self.queue_total_waiting_first_record, 
                average_btt_list, self.get_file_path(test_config, 'intensity_bt_latency.png'))
            trace_plot_intensity_with_analysis_data(self.queue_total_submitted_record, self.queue_total_waiting_first_record, 
                average_ftt_list, self.get_file_path(test_config, 'intensity_ft_latency.png'))
            trace_plot_aggregated_intensity(aggregated_intensity, self.get_file_path(test_config, 'aggregated.png'))
            trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, average_btt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'aggregated_bt_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, average_ftt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'aggregated_ft_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, throughput_list,
                ('Throughput', ''), self.get_file_path(test_config, 'aggregated_throughput.png'))
    
    def check_performace(self, test_config):
        log_filepath = os.path.join(self.log_path, "-".join([test_config.tag, "all.log"]))
        return self.total_submitted * 0.01 >= self.failed, self.completed, self.total_submitted - self.failed, self.total_submitted, (self.first_token_slo, self.idle_time_slo)

    # Entering into this function automatically indicates is under statistics recording mode
    def check_single_query_performance(self, request_timestamp, first_chunk_time, user_idle_time, last_received_time, reading_time):
        if(request_timestamp == None):
            return
        failed_flag = False
        self.total_submitted += 1
        current_timestamp = time.perf_counter() - self.start_time_pointer
        self.time_intervals.append((request_timestamp, current_timestamp))
        if(first_chunk_time == None):
            first_chunk_time = current_timestamp - request_timestamp
        if(last_received_time != None):
            user_idle_time = max(user_idle_time, current_timestamp - last_received_time - reading_time)
        failed_flag = (first_chunk_time > self.first_token_slo)
        failed_flag = failed_flag or (user_idle_time > self.idle_time_slo)
        if(failed_flag):
            self.failed += 1
    
    async def _inference(self, input, rec, max_tokens):
        user_idle_time = 0
        first_chunk_time = 0
        reading_time = 0
        waiting_time = 0
        request_timestamp = None
        last_received_time = None
        current_timestamp = None
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
                current_timestamp = time.perf_counter() - self.start_time_pointer
                self.recording_queue_records(rec, 0, self.queue_total_submitted_record, current_timestamp, 1)
                self.recording_queue_records(rec, 1, self.queue_total_waiting_first_record, current_timestamp, 1)
                reading_time = 0
                request_timestamp = time.perf_counter() - self.start_time_pointer
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
                            current_timestamp = time.perf_counter() - self.start_time_pointer
                            if not first_chunk_received:
                                self.throughput_counter += 1
                                first_chunk_time = current_timestamp - request_timestamp
                                first_chunk_received = True
                                if(rec): 
                                    self.time_bins[int(current_timestamp / BIN_SIZE)][0].append(first_chunk_time)
                                self.recording_queue_records(rec, 1, self.queue_total_waiting_first_record, current_timestamp, -1)
                            else:
                                between_token_time = current_timestamp - last_received_time
                                waiting_time += between_token_time
                                if(rec): 
                                    self.time_bins[int(current_timestamp / BIN_SIZE)][1].append(between_token_time)
                                user_idle_time = max(user_idle_time, waiting_time - reading_time)
                                reading_time += self.SPEED_CONST
                            last_received_time = current_timestamp
                        # output = json.loads(buffer.decode("utf-8"))  # Decode JSON
                    else:
                        output = await resp.json() 
            self.recording_queue_records(rec, 0, self.queue_total_submitted_record, current_timestamp, -1) 
            if(rec):
                self.completed += 1
                self.logs.append([request_timestamp, first_chunk_time, current_timestamp - request_timestamp, user_idle_time])
                self.check_single_query_performance(request_timestamp, first_chunk_time, user_idle_time, last_received_time, reading_time)
        except asyncio.CancelledError:
            if(rec):
                self.check_single_query_performance(request_timestamp, first_chunk_time, user_idle_time, last_received_time, reading_time)
            raise

    def inference(self, input, rec = False, max_tokens = 15):
        self.loop.run_until_complete(self._inference(input, rec, max_tokens))

    def random_test_period(self, test_config, time_len, rec = RecordingAction.NO_RECORD):
        self.start_time_pointer = time.perf_counter()
        if(test_config.seq_len == -1):
            test_config.seq_len = [10, 10]
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        while(time.perf_counter() - self.start_time_pointer < time_len):
            input = self.get_input(test_config)
            self.inference(input, rec in RECORDING_ACTION_MAP["statistics"])
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)

    async def intensity_monitor_udp(self):
        print("Starting UDP server")
        self.intensity_server = None
        self.intensity_server, _ = await self.loop.create_datagram_endpoint(
            lambda: UDPProtocol(self.queue_record),
            local_addr=('127.0.0.1', LLM_SERVER_PORT)
        )
    
    async def unix_handler(self, reader, writer):
        data = await reader.read(32)
        message = data.decode()
        writer.write(self.pack_scheduler_data(self))
        await writer.drain()
        writer.close()
    
    async def intensity_monitor_unix(self):
        print("Starting Unix server")
        self.intensity_server = None
        self.intensity_server = await asyncio.start_unix_server(self.unix_handler, LLM_SOCKET_PATH)
    
    async def throughput_monitor(self):
        try:
            while(True):
                self.record_throughput = self.throughput_counter
                self.throughput_counter = 0
                self.record_intensity = self.record_intensity_prev
                self.record_intensity_prev = self.queue_record[1]
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.throughput_counter = 0
            self.record_throughput = 0
            self.record_intensity_prev = 0
            raise

    # rec: true or false
    async def trace_test_period(self, test_config, time_len, rec):
        if(rec and self.time_bins == None):
            # first list for first_token_time and second list for time_between_token
            self.time_bins = [[[],[]] for x in np.arange(0, time_len + BIN_SIZE * 3, BIN_SIZE)]
        time_limit = self.start_time_pointer + time_len
        asyncio.create_task(self.intensity_monitor_unix())
        asyncio.create_task(self.throughput_monitor())

        task_list = []
        while(True):
            req_time, sample_in_length, sample_out_length = self.trace.next()
            test_config.seq_len = [sample_in_length, sample_out_length]
            sleep_time = req_time + self.start_time_pointer - time.perf_counter()
            if(sleep_time > 0):
                await asyncio.sleep(sleep_time)
            if(time.perf_counter() > time_limit):
                break
            input = self.get_input(test_config)
            task_list.append(asyncio.create_task(self._inference(input, 
                rec, sample_out_length)))
        
        # await asyncio.gather(*task_list)
        for id, task in enumerate(task_list):
            if(not task.done()):
                task.cancel()
        if(self.intensity_server != None):
            self.intensity_server.close()
            print("intensity server stoped")
        # print("processed:", cnt, "in", time.perf_counter() - test_start_time, "(s) uncompleted:", uncompleted)
    
    def common_start(self):
        self.total_module = -1
    
    def random_test(self, test_config):
        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        recording_action = RecordingAction.ALL if test_config.recording_action == None else test_config.recording_action
        if(test_config.scheduler != None):
            global schedulers
            self.pack_scheduler_data = schedulers[test_config.scheduler].pack_func
        
        if(recording_action in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        
        self.start_time_pointer = time.perf_counter()
        if(test_config.trace_path != None):
            self.loop.run_until_complete(self.trace_test_period(test_config, test_time, recording_action))
        else:
            self.random_test_period(test_config, test_time, recording_action)

        if(recording_action in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)

        if(recording_action in RECORDING_ACTION_MAP["statistics"]):
            self.write_log(test_config)
    

