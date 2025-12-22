from profiling.models.model_worker import ModelWorker
from dataset.new_sharegpt_dataset import NewShareGPTDataset
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

async def get_chunk_with_timeout(receiver):
    timeout_setting = 0.93
    try:
        chunk = await asyncio.wait_for(receiver.__anext__(), timeout=timeout_setting)  # Timeout per iteration
    except asyncio.TimeoutError:
        chunk = await receiver.__anext__()
    except StopAsyncIteration:
        return None

async def get_chunk(receiver):
    try:
        chunk = await receiver.__anext__()
        return chunk
    except StopAsyncIteration:
        return None

class LLMWorker(ModelWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("llm", device_id, mps, pipe, barrier, log_path, profile_path)
        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True
        }
        self.SPEED_CONST = 0.2
        self.initial_time = 0.2
        self.server_proc = None
        self.time_bins = None
        self.first_token_slo = 1
        self.idle_time_slo = 0
        self.queue_total_submitted_record = []
        self.queue_total_waiting_first_record = []
        self.intensity_server = None
        self.intensity_profiler = None
        self.scheduler_monitor = None
        self.device = device_id
        
    def recording_queue_records(self, rec, target_id, target_list, timestamp, value):
        self.queue_record[target_id] += value
        if(rec != False and timestamp < self.time_limit):
            target_list.append((timestamp, self.queue_record[target_id]))
    
    def statistics_recording_init(self, test_config):
        time_len = self.test_time if test_config.test_time == None else test_config.test_time
        self.time_bins = [[[],[]] for x in timebin_ranges(PLOTTING_BIN_SIZE, time_len)]
        self.time_bins_failed_record = [0 for x in timebin_ranges(PLOTTING_BIN_SIZE, time_len)]
        self.logs = []
        # total_submitted & waiting_first
        self.queue_total_submitted_record = []
        self.queue_total_waiting_first_record = []
    
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
        return test_config.seq_len[0], self.dataset.get_query(test_config)[0]
    
    def get_trace(self, trace_path = None):
        self.trace = get_LLM_trace()
    
    def get_tag(self, test_config):
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
    
    def write_basic_logs(self, test_config):
        self.write_analytics(test_config, "ftsummary.log", [x[3] for x in self.logs])
        self.write_analytics(test_config, "btsummary.log", [(x[4]/ x[2]) for x in self.logs if(x[4] > 0)])
        self.logs = sorted(self.logs, key = lambda x: x[0])
        self.write_list_to_csv(test_config, "all.csv", 
            ['Request timestamp (s)', 'Input token cnt', 'Output token cnt', 'TFT (s)', 'Latency (s)', 'Idle time (s)', 'Success'], 
            self.logs)
    
    def write_log(self, test_config):
        self.write_basic_logs(test_config)
        average_ftt_list = []
        average_btt_list = []
        throughput_list = []
        tt_logs = []
        pointer_first = 0
        pointer_all = 0
        first_intensity = 0
        all_intensity = 0
        for id, bin in enumerate(self.time_bins): 
            bin_start = id * PLOTTING_BIN_SIZE
            while(pointer_first < len(self.queue_total_waiting_first_record) 
                and self.queue_total_waiting_first_record[pointer_first][0] < bin_start + 0.02):
                first_intensity = self.queue_total_waiting_first_record[pointer_first][1]
                pointer_first += 1
            while(pointer_all < len(self.queue_total_submitted_record) 
                and self.queue_total_submitted_record[pointer_all][0] < bin_start + 0.02):
                all_intensity = self.queue_total_submitted_record[pointer_all][1]
                pointer_all += 1
            average_ftt = np.average(bin[0]) if len(bin[0]) > 0 else 0
            max_ftt = np.max(bin[0]) if len(bin[0]) > 0 else 0
            average_btt = np.average(bin[1]) if len(bin[1]) > 0 else 0
            max_btt = np.max(bin[1]) if len(bin[1]) > 0 else 0
            average_ftt_list.append(average_ftt)
            average_btt_list.append(average_btt)
            throughput_list.append(len(bin[1]) + len(bin[0]))
            tt_logs.append([bin_start, first_intensity, all_intensity - first_intensity, len(bin[0]), len(bin[1]), average_ftt, max_ftt, average_btt, max_btt])
        self.write_list_to_csv(test_config, "ttsummary.csv", 
            ['Bin Timestamp (s)', 'Prefill intensity', 'Decode intensity', 'FT throughput', 'BT throughput', 'Average TFT (s)', 'Max TFT (s)', 'Average TBT (s)', 'Max TBT(s)'], 
            tt_logs)
        
        average_ftt_list = bin_records_append_timestamps(average_ftt_list, PLOTTING_BIN_SIZE)
        average_btt_list = bin_records_append_timestamps(average_btt_list, PLOTTING_BIN_SIZE)
        throughput_list = bin_records_append_timestamps(throughput_list, PLOTTING_BIN_SIZE)
        failed_record_list = bin_records_append_timestamps(self.time_bins_failed_record, PLOTTING_BIN_SIZE)

        aggregated_intensity = bin_data(self.queue_total_submitted_record)

        if(test_config.trace_path != None):
            self.write_all(test_config, "aggregated_intensity_all.log", aggregated_intensity)
            self.write_all(test_config, "failed_records.log", failed_record_list)
            trace_plot_two_intensity_with_analysis_data(self.queue_total_submitted_record, "All", self.queue_total_waiting_first_record, "First",
                average_btt_list, "Average BTT", ('Latency', '(s)'), self.get_file_path(test_config, 'intensity_bt_latency.png'))
            trace_plot_two_intensity_with_analysis_data(self.queue_total_submitted_record, "All", self.queue_total_waiting_first_record, "First",
                average_ftt_list, "Average FTT", ('Latency', '(s)'), self.get_file_path(test_config, 'intensity_ft_latency.png'))
            trace_plot_two_intensity_with_analysis_data_bar_plot(self.queue_total_submitted_record, "All", self.queue_total_waiting_first_record, "First",
                failed_record_list, "Failed", ('Case Count', ''), self.get_file_path(test_config, 'aggregated_failed.png'))
            trace_plot_aggregated_intensity(aggregated_intensity, self.get_file_path(test_config, 'aggregated.png'))
            trace_plot_aggregated_intensity_with_analysis_data(bin_data(self.queue_total_waiting_first_record), average_btt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'prefill_bt_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, average_btt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'aggregated_bt_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, average_ftt_list,
                ('Latency', '(s)'), self.get_file_path(test_config, 'aggregated_ft_latency.png'))
            trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, throughput_list,
                ('Throughput', ''), self.get_file_path(test_config, 'aggregated_throughput.png'))
    
    def check_performace(self, test_config):
        return self.total_submitted * 0.01 >= self.failed, self.completed, self.total_submitted - self.failed, self.total_submitted, (self.first_token_slo, self.idle_time_slo, self.SPEED_CONST, self.initial_time)

    # Return True: meet slo; False: violated slo
    def check_single_query_performance(self, request_timestamp, first_chunk_time, user_idle_info, rec):
        user_idle_time, user_idle_time_info_record = user_idle_info
        if(request_timestamp == None):
            return True
        self.total_submitted += 1
        if(first_chunk_time > self.first_token_slo):
            self.failed += 1
            if(rec):
                self.time_bins_failed_record[int((request_timestamp + self.first_token_slo) / PLOTTING_BIN_SIZE)] += 1
            return False
        else:
            if(user_idle_time_info_record != None):
                self.failed += 1
                if(rec):
                    self.time_bins_failed_record[int(user_idle_time_info_record / PLOTTING_BIN_SIZE)] += 1
                return False
        return True

    async def stream_response(self, input_length, output_length, rec, receiver, request_timestamp):
        
        def update_user_idle_time_info_record(user_idle_time_info_record, user_idle_time, idle_time, idle_slo):
            if(idle_time > user_idle_time):
                user_idle_time = idle_time
                if(user_idle_time_info_record == None and idle_time > idle_slo):
                        user_idle_time_info_record = current_timestamp - user_idle_time + idle_slo
            return user_idle_time_info_record, user_idle_time

        first_chunk_time = None
        user_idle_time = 0
        user_idle_time_info_record = None
        last_received_time = request_timestamp
        reading_time = self.initial_time
        waiting_time = 0
        
        token_cnt = input_length
        decode_flag = False
        output_cnt = 0

        received_token_tmp = 0
        last_refreshed_flag = self.refreshed_flag
        try:
            async for chunk in receiver:
                # If this is the first chunk, record the time taken
                current_timestamp = time.perf_counter() - self.start_time_pointer
                if(current_timestamp > self.time_limit):
                    raise asyncio.CancelledError
                current_plotting_bin_timestamp = int(current_timestamp / PLOTTING_BIN_SIZE)
                between_token_time = current_timestamp - last_received_time
                token_cnt += 1
                self.uncompleted_tokens -= 1
                output_cnt += 1

                if first_chunk_time == None:
                    first_chunk_time = between_token_time
                    if(rec): 
                        self.time_bins[current_plotting_bin_timestamp][0].append(first_chunk_time)
                    self.recording_queue_records(rec, 1, self.queue_total_waiting_first_record, current_timestamp, -1)
                    self.ftt_cnt_rt += 1
                    self.total_decode_length += token_cnt
                    self.total_prefill_length -= input_length
                    self.prefill_start_time -= request_timestamp
                    self.prefill_length_heap.delete((input_length, request_timestamp))
                    self.decode_intensity += 1
                    decode_flag = True
                else:
                    if(last_refreshed_flag != self.refreshed_flag):
                        received_token_tmp = 0
                        last_refreshed_flag = self.refreshed_flag
                    received_token_tmp += 1
                    waiting_time += between_token_time
                    if(rec):
                        self.time_bins[current_plotting_bin_timestamp][1].append(between_token_time)
                    self.total_btt_rt += between_token_time
                    self.btt_cnt_rt += 1

                    self.total_decode_length += 1
                    
                    user_idle_time_info_record, user_idle_time = update_user_idle_time_info_record(
                        user_idle_time_info_record, user_idle_time, waiting_time - reading_time, self.idle_time_slo)
                    reading_time += self.SPEED_CONST
                last_received_time = current_timestamp
            self.total_decode_length -= token_cnt
            if(last_refreshed_flag == self.refreshed_flag):
                self.finished_token_tmp += received_token_tmp
                self.finished_tmp += 1
            if(decode_flag):
                self.decode_intensity -= 1
                decode_flag = False
            if(output_length != output_cnt):
                self.uncompleted_tokens -= output_length - output_cnt
            return first_chunk_time, user_idle_time, user_idle_time_info_record, 1
        except asyncio.CancelledError:
            current_timestamp = time.perf_counter() - self.start_time_pointer
            between_token_time = current_timestamp - last_received_time
            if(first_chunk_time != None):
                waiting_time += between_token_time
                user_idle_time_info_record, user_idle_time = update_user_idle_time_info_record(
                    user_idle_time_info_record, user_idle_time, waiting_time - reading_time, self.idle_time_slo)
            else:
                first_chunk_time = between_token_time
            if(decode_flag):
                self.decode_intensity -= 1
                decode_flag = False
            return first_chunk_time, user_idle_time, user_idle_time_info_record, 0

    async def _inference(self, input, rec, max_tokens):
        first_chunk_time = None 
        user_idle_time = 0 
        user_idle_time_info_record = None
        request_timestamp = None
        self.uncompleted_tokens += max_tokens
        completed = False
        try:
            # no start & end available for LLMworker served by vllm
            timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
            current_timestamp = time.perf_counter() - self.start_time_pointer
            self.recording_queue_records(rec, 0, self.queue_total_submitted_record, current_timestamp, 1)
            self.recording_queue_records(rec, 1, self.queue_total_waiting_first_record, current_timestamp, 1)
            request_timestamp = time.perf_counter() - self.start_time_pointer

            self.prefill_start_time += request_timestamp
            self.prefill_start_time_tmp += request_timestamp                        
            self.total_prefill_length += input[0]
            self.total_prefill_length_tmp += input[0]
            self.prefill_length_heap.push((input[0], request_timestamp))
            heapq.heappush(self.prefill_length_heap_tmp, (-input[0], request_timestamp))
            self.prefill_intensity_tmp += 1

            async with aiohttp.ClientSession(timeout=timeout) as session:
                generation_input = {
                    "prompt": input[1],
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
                        self.stream_response(input[0], max_tokens, rec, resp.content.iter_any(), request_timestamp)
                    # output = json.loads(buffer.decode("utf-8"))  # Decode JSON
            current_timestamp = time.perf_counter() - self.start_time_pointer

            self.recording_queue_records(rec, 0, self.queue_total_submitted_record, current_timestamp, -1) 
            if(completed):
                self.completed += 1
                latency = current_timestamp - request_timestamp
            else:
                latency = -1
            result = self.check_single_query_performance(request_timestamp, first_chunk_time, (user_idle_time, user_idle_time_info_record), rec)
            if(rec):
                self.logs.append([request_timestamp, input[0], max_tokens, first_chunk_time, latency, user_idle_time, result])
        except asyncio.CancelledError:
            raise

    def inference(self, input, rec = False, max_tokens = 15):
        self.loop.run_until_complete(self._inference(input, rec, max_tokens))
    
    async def simple_stream_response(self, receiver, request_timestamp):
        bt_recs = []
        last_received_time = request_timestamp
        async for chunk in receiver:
            # If this is the first chunk, record the time taken
            current_timestamp = time.perf_counter() - self.start_time_pointer
            between_token_time = current_timestamp - last_received_time
            bt_recs.append(between_token_time)
            last_received_time = current_timestamp
        return bt_recs

    async def simple_inference(self, input, max_tokens = 15):
        first_chunk_time = None 
        user_idle_time = 0 
        user_idle_time_info_record = None
        request_timestamp = None
        # no start & end available for LLMworker served by vllm
        timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
        request_timestamp = time.perf_counter() - self.start_time_pointer
        async with aiohttp.ClientSession(timeout=timeout) as session:
            generation_input = {
                "prompt": input[1],
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
                bt_recs = await self.simple_stream_response(resp.content.iter_any(), request_timestamp)
        return bt_recs
        # self.logs.append([request_timestamp, input[0], max_tokens, first_chunk_time, current_timestamp - request_timestamp, user_idle_time])

    def random_test_period(self, test_config, time_len, rec = False):
        self.start_time_pointer = time.perf_counter()
        self.completed = 0
        self.failed = 0
        self.total_submitted = 0
        self.monitor_init(test_config)
        if(test_config.seq_len == None):
            test_config.seq_len = [10, 10]
        self.time_limit = time_len
        time_limit = self.start_time_pointer + time_len
        while(time.perf_counter() < time_limit):
            input = self.get_input(test_config)
            self.inference(input, rec,  test_config.seq_len[1])
    
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

    # async def intensity_monitor_udp(self):
    #     print("Starting UDP server")
    #     self.intensity_server = None
    #     self.intensity_server, _ = await self.loop.create_datagram_endpoint(
    #         lambda: UDPProtocol(self.queue_record),
    #         local_addr=('127.0.0.1', LLM_SERVER_PORT)
    #     )
    
    async def unix_handler(self, reader, writer):
        _ = await reader.read(32)
        data = self.scheduling_data_format.pack_data(self)
        writer.write(len(data).to_bytes(4, "big"))
        await writer.drain()
        writer.write(data)
        await writer.drain()
        writer.close()
    
    async def intensity_server_unix(self):
        print("Starting Unix server")
        self.intensity_server = None
        self.intensity_server = await asyncio.start_unix_server(self.unix_handler, LLM_SOCKET_PATH)
    
    def monitor_init(self, test_config):
        time_len = self.test_time if test_config.test_time == None else test_config.test_time
        # all_cnt, prefill_cnt ::::: Note the previous bug here! previous decode cnt, prefill_cnt (wrong)
        self.queue_record = [0, 0]

        # advanced_aimd: total_btt_rt: total btt; btt_cnt_rt: bt throughput; ftt_cnt_rt: ft throughput
        # Only adding here, cleared on query
        # btt_cnt_rt & ftt_cnt_rt also used by $predict$
        self.total_btt_rt = 0
        self.btt_cnt_rt = 0
        self.ftt_cnt_rt = 0

        # basic_aimd
        self.uncompleted_tokens = 0

        # predict
        self.total_prefill_length = 0
        self.total_decode_length = 0
        self.prefill_intensity_tmp = 0
        self.total_prefill_length_tmp = 0 # Same as total_prefill_length but cleared on query
        self.prefill_start_time = 0 # Sum of all request start time
        self.prefill_start_time_tmp = 0 # Same as prefill_start_time but cleared on query
        self.prefill_length_heap_tmp = []
        self.prefill_length_heap = DeletableHeap()
        self.decode_intensity = 0
        self.finished_tmp = 0
        self.refreshed_flag = 0
        self.finished_token_tmp = 0

    # rec: true or false
    async def trace_test_period(self, test_config, time_len, rec = False):
        self.monitor_init(test_config)
        if(self.test_config.global_cnt > 1 and test_config.scheduler != None): # Serving mode
            asyncio.create_task(self.intensity_server_unix())
            scheduler_type = type(get_scheduler(test_config.scheduler))
            if(scheduler_type.monitor != None):
                self.scheduler_monitor = asyncio.create_task(scheduler_type.monitor(self))
        elif (test_config.scheduler != None): # Profiling mode
            print("scheduler profiling")
            scheduler_type = type(get_scheduler(test_config.scheduler))
            self.intensity_profiler = asyncio.create_task(scheduler_type.profiling_executor(self, test_config))
        max_in_tokens = 0
        min_in_tokens = 10000000
        max_out_tokens = 0
        min_out_tokens = 10000000
        task_list = []

        if(test_config.scheduler != None and self.test_config.global_cnt > 1):
            self.double_barrier.wait()
        self.start_time_pointer = time.perf_counter()
        time_counter = time_len
        while(True):
            req_time, sample_in_length, sample_out_length = self.trace.next()
            max_in_tokens = max(max_in_tokens, sample_in_length)
            min_in_tokens = min(min_in_tokens, sample_in_length)
            max_out_tokens = max(max_out_tokens, sample_out_length)
            min_out_tokens = min(min_out_tokens, sample_out_length)
            test_config.seq_len = [sample_in_length, sample_out_length]
            sleep_time = req_time + self.start_time_pointer - time.perf_counter()
            if(time_counter - sleep_time > 0):
                if(sleep_time > 0):
                    await asyncio.sleep(sleep_time)
                    time_counter -= sleep_time
            else: 
                break
            input = self.get_input(test_config)
            task_list.append(asyncio.create_task(self._inference(input, 
                rec, sample_out_length)))
        await asyncio.sleep(time_counter)
        # await asyncio.gather(*task_list)
        if(self.intensity_server != None):
            self.intensity_server.close()
            print("intensity server stoped")
        if(self.intensity_profiler != None):
            self.intensity_profiler.cancel()
        if(self.scheduler_monitor != None):
            self.scheduler_monitor.cancel()
        for id, task in enumerate(task_list):
            if(not task.done()):
                task.cancel()
        print("max request tokens: ", max_in_tokens, "; min request tokens", min_in_tokens)
        print("max response tokens: ", max_out_tokens, "; min response tokens", min_out_tokens)

        # print("processed:", cnt, "in", time.perf_counter() - test_start_time, "(s) uncompleted:", uncompleted)
    
    def submodule_setup(self):
        self.total_module = -1
    
    def random_test(self, test_config):
        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        self.time_limit = test_time
        recording_action = RecordingAction.ALL if test_config.recording_action == None else test_config.recording_action
        self.monitor_init(test_config)
        self.completed = 0
        self.failed = 0
        self.total_submitted = 0

        if(test_config.scheduler != None):
            self.scheduling_data_format = type(get_scheduler(test_config.scheduler)).data_format
        
        record_statistics = (recording_action in RECORDING_ACTION_MAP["statistics"])
        if(record_statistics):
            self.statistics_recording_init(test_config)

        if(recording_action in RECORDING_ACTION_MAP["dcgm"]):
            dcgm_monitor_conn = self.start_dcgm(test_config)
        
        if(test_config.trace_path != None):
            self.loop.run_until_complete(self.trace_test_period(test_config, test_time, record_statistics))
        else:
            self.random_test_period(test_config, test_time, record_statistics)

        if(recording_action in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(dcgm_monitor_conn)

        if(record_statistics):
            self.write_log(test_config)
    

