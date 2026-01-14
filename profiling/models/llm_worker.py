import requests
import subprocess
import time
import signal
import os
import asyncio
import aiohttp
from profiling.models.model_worker import ModelWorker
from profiling.constants import *
from profiling.scheduling.monitor import LLM_SOCKET_PATH
from profiling.scheduling.utils import get_scheduler_class
from profiling.scheduling import AIMDScheduler, AIMDSchedulingDataFormat, PredictiveScheduler, PredictiveSchedulingDataFormat
from dataset.sharegpt_dataset import ShareGPTDataset
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config
from plotting.trace_plot import TracePlotData, TraceAxisData, plot_trace_graph

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
        memory_util = os.environ.get('GVLLM_KV_MEM', "0.7")
        self.server_proc = subprocess.Popen(["python", "-m", "vllm.entrypoints.api_server"
            , "--model", "meta-llama/Llama-2-7b-chat-hf", "--gpu-memory-utilization", memory_util])
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
        if(self.server_proc != None):
            self.server_proc.send_signal(signal.SIGINT)
    
    """
        Get input config of llm worker from test config.
        Test config's seq_len: [prompt_len, output_len], only prompt_len is used for input.
        Default value is set in InputConfig's init instead of here (here: None).
    """
    @classmethod
    def get_input_config_from_test_config(cls, test_config):
        return InputConfig(seq_len = test_config.seq_len[0] if test_config.seq_len else None)

    def get_input(self, input_config):
        prompt, prompt_len = self.dataset.get_query(input_config)
        return prompt, prompt_len
    
    def get_trace_config(self, test_config):
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
    
    """
        Get the tag of oneself in the format of:
            model_name(10,10) (* 10,10 is seq_len)
            model_name(10,10_m70)
            model_name(m30)[trace_tag]
            model_name[trace_tag]
    """
    def get_self_tag(self, test_config):
        trace_config = self.get_trace_config(test_config)
        tag = self.model_name
        # Ignore seq_len if trace_path is set.
        if(test_config.seq_len and (test_config.trace_path is None)):
            subtag = ",".join(str(x) for x in test_config.seq_len)
        else:
            subtag = ""
        if(self.mps is not None and self.mps != 100):
            if(subtag):
                subtag += "_"
            subtag += "m" + str(self.mps)
        if(subtag):
            tag += "(" + subtag + ")"
        if(test_config.trace_path):
            tag += "[" + trace_config.get_trace_tag() + "]"
        return tag
    
    """
        Perform inferences for GPU & model warmup.
    """
    def warmup(self):
        warmup_iter = 10
        llm_input = self.get_input(InputConfig(seq_len = 10))
        self.runtime_data.start_time_pointer = time.perf_counter()
        for _ in range(warmup_iter):
            self.loop.run_until_complete(self.async_inference(llm_input, 10))
    
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
        count_list_ftt, avg_list_ftt, max_list_ftt = self.profile_data.first_token_time.collapse()
        count_list_btt, avg_list_btt, max_list_btt = self.profile_data.between_token_time.collapse()

        tt_logs = [
            [id * self.profile_data.first_token_time.bin_size, *values]
            for id, values in enumerate(zip(
                count_list_ftt, count_list_btt, avg_list_ftt,
                max_list_ftt, avg_list_btt, max_list_btt
            ))
        ]
        self.write_list_to_csv("ttsummary.csv", 
            ['Bin Timestamp (s)', 'FT throughput', 'BT throughput', 'Average TFT (s)', 'Max TFT (s)', 'Average TBT (s)', 'Max TBT(s)'], 
            tt_logs)
        
        if(self.test_config.trace_path):
            _, avg_submitted, _ = self.runtime_data.submitted_record_counter.record_bins.collapse()
            avg_submitted_intensity = self.runtime_data.submitted_record_counter.record_bins.append_timestamps(avg_submitted)
            failed_records = self.profile_data.failed_records.append_timestamps()
            self.write_all("avg_submitted_intensity.log", avg_submitted_intensity)
            self.write_all("failed_records.log", failed_records)

            avg_ftt_axis_data = TraceAxisData(TracePlotData('Avg First token time', self.profile_data.first_token_time.append_timestamps(avg_list_ftt)), y_tag = ('Latency', 's'))
            avg_btt_axis_data = TraceAxisData(TracePlotData('Avg Between token time', self.profile_data.between_token_time.append_timestamps(avg_list_btt)), y_tag = ('Latency', 's'))
            failed_records_axis_data = TraceAxisData(TracePlotData('Failed req count', failed_records), y_tag = ('Count', ''), plot_type = TraceAxisData.PlotType.BAR)
            _, avg_waiting_first_intensity, _ = self.runtime_data.waiting_first_record_counter.record_bins.collapse()
            avg_submitted_intensity = TracePlotData('Num of submitted reqs', avg_submitted_intensity)
            avg_waiting_first_intensity = TracePlotData('Num of reqs waiting for first token', self.runtime_data.waiting_first_record_counter.record_bins.append_timestamps(avg_waiting_first_intensity))
            
            intensity_axis_data = TraceAxisData(avg_submitted_intensity, avg_waiting_first_intensity, label = 'Intensity', y_tag = ('Intensity',''))
            plot_trace_graph(self.get_file_path('intensity_bt_latency.png'), intensity_axis_data, avg_btt_axis_data)
            plot_trace_graph(self.get_file_path('intensity_ft_latency.png'), intensity_axis_data, avg_ftt_axis_data)
            plot_trace_graph(self.get_file_path('intensity_failed.png'), intensity_axis_data, failed_records_axis_data)
            
    
    """
        Get overall performance statistics of previous test run:
            passed_flag (99% SLO), completed reqs, passed reqs, total submitted
        Attached is SLO information:
            (ftt slo, btt slo, reading speed, initial time (reading time set for first token))
    """
    def check_performace(self):
        passed_flag = self.runtime_data.total_submitted * 0.01 >= self.runtime_data.failed
        passed_count = self.runtime_data.total_submitted - self.runtime_data.failed
        return (passed_flag, self.runtime_data.completed, passed_count, self.runtime_data.total_submitted, 
            (self.first_token_slo, self.idle_time_slo, self.SPEED_CONST, self.initial_time))

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

                if first_chunk_time is None:
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
    
    """
        FIXME: Haven't decided what to do with LLM profiling.
    """
    def profile(self):
        pass
    
    async def unix_handler(self, reader, writer):
        _ = await reader.read(32)
        data = self.scheduler_data.scheduling_data_format.pack_data(self)
        writer.write(len(data).to_bytes(4, "big"))
        await writer.drain()
        writer.write(data)
        await writer.drain()
        writer.close()
    
    async def create_unix_intensity_server(self):
        print("Starting Unix server")
        self.scheduler_data.intensity_server = await asyncio.start_unix_server(self.unix_handler, LLM_SOCKET_PATH)
    
    async def _test(self):
        if(self.test_config.scheduler):
            if(self.test_config.global_cnt > 1): # Serving mode
                asyncio.create_task(self.create_unix_intensity_server())
                if(self.scheduler_data.scheduler_type.monitor):
                    self.scheduler_data.scheduler_monitor = asyncio.create_task(self.scheduler_data.scheduler_type.monitor(self))
            else: # Profiling mode
                print("Scheduler profiling...")
                self.scheduler_data.scheduler_profiler = asyncio.create_task(self.scheduler_data.scheduler_type.profiling_executor(self))
                
        task_list = []
        if(self.test_config.scheduler != None and self.test_config.global_cnt > 1):
            self.double_barrier.wait()
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time

        while(time.perf_counter() < self.runtime_data.time_limit):
            if(self.test_config.trace_path):
                req_time, sample_in_length, sample_out_length = self.test_config.trace.next()

                if(req_time < self.test_config.test_time):
                    sleep_time = req_time + self.runtime_data.start_time_pointer - time.perf_counter()
                    if(sleep_time > 0):
                        await asyncio.sleep(sleep_time)
                else: 
                    break
                llm_input = self.get_input(InputConfig(seq_len = sample_in_length))
                task_list.append(asyncio.create_task(self.async_inference(llm_input, sample_out_length)))
            else:
                llm_input = self.get_input(self.input_config)
                await self.async_inference(llm_input, self.test_config.seq_len[1])

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

