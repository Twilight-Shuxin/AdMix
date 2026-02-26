import time
import torch.multiprocessing as mp
from profiling.constants import *
from profiling.models.workers import LLMWorker
from profiling.models.model_client import ModelClient
from profiling.models.workers.constants import SyncType
from plotting.trace_plot import TracePlotData, TraceAxisData, plot_trace_graph

class LLMClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path):
        self.first_token_slo = 1
        self.idle_time_slo = 0
        self.SPEED_CONST = 0.2
        self.initial_time = self.SPEED_CONST
        self.model_worker = LLMWorker
        super().__init__("llm", devices, mps, pipe, barrier, log_path, profile_path)

    def start_worker(self, devices):
        device_init(self.devices, devices, self.mps)
        pipe_parent, pipe_child = mp.Pipe()
        worker = self.model_worker(pipe_child, self.worker_barrier, self.single_worker_barrier, 
            self.log_path, self.profile_path,
            (self.first_token_slo, self.idle_time_slo, self.SPEED_CONST))
        worker.start()
        self.worker_pipes.append(pipe_parent)

    def start_workers(self):
        # NOTE: LLM client has only one worker, so both worker_barrier & single_worker_barrier expects two participants.
        # Both is keeped to sustain compatibility with ModelClient base pattern.
        self.worker_barrier = mp.Barrier(2)
        self.single_worker_barrier = mp.Barrier(2)
        self.start_worker(self.devices)
        self.worker_barrier.wait()
    
    """
        Get the tag of oneself in the format of:
            model_name(10,10) (* 10,10 is seq_len)
            model_name(10,10_m70)
            model_name(m30)[trace_tag]
            model_name[trace_tag]
    """
    def get_self_tag(self, test_config = None):
        if(not test_config):
            test_config = self.test_config
        trace_config = LLMWorker.get_trace_config(test_config)
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
        Write basic logs:
            profile_data.logs: [(timestamp, prompt_tkn, output_tkn, ttft, latency, idle time, success)]
    """
    def write_basic_logs(self):
        self.profile_data.logs = sorted(self.profile_data.logs, key = lambda x: x[0])
        self.write_analytics("ftsummary.log", [x[3] for x in self.profile_data.logs])
        self.write_analytics("btsummary.log", [(x[4] / x[2]) for x in self.profile_data.logs])
        self.write_csv("all.csv", 
            ['Request timestamp (s)', 'Input token cnt', 'Output token cnt', 'TFT (s)', 'Latency (s)', 'Idle time (s)', 'Success'], 
            self.profile_data.logs)
    
    def write_logs(self):
        self.write_basic_logs()
        count_list_ftt, avg_list_ftt, max_list_ftt = self.profile_data.first_token_time.collapse()
        count_list_btt, avg_list_btt, max_list_btt = self.profile_data.between_token_time.collapse()

        tt_logs = [
            [idx * self.profile_data.first_token_time.bin_size, *values]
            for idx, values in enumerate(zip(
                count_list_ftt, count_list_btt, avg_list_ftt,
                max_list_ftt, avg_list_btt, max_list_btt
            ))
        ]
        self.write_csv("ttsummary.csv", 
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
        FIXME: Haven't decided what to do with LLM profiling.
    """
    def profile(self):
        pass

    """
        Currently, a llm client only has a single worker (which communicates with llm server with tensor parallelism).
        So collection of worker data does not have combining functionality, only retrieving data from the single worker.
    """
    def collect_worker_data(self):
        for idx, pipe in enumerate(self.worker_pipes):
            self.command_worker(pipe, "collect_performance_data")
            self.runtime_data = pipe.recv()

        if(self.test_config.record_statistics):
            for idx, pipe in enumerate(self.worker_pipes):
                self.command_worker(pipe, "collect_profiled_data")
                self.profile_data = pipe.recv()
    
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
    
    def _test(self):
        self.command_workers("test", sync = SyncType.ALL)