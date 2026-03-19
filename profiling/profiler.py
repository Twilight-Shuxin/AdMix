from profiling.utils import model_clients
from profiling.constants import *
import torch.multiprocessing as mp
from plotting.trace_plot import TracePlotData, TraceAxisData, plot_trace_graph
from trace.trace_utils import set_LLM_trace_config
import random
import subprocess 
import time
import os

class ProfileServer():
    def __init__(self, profile_config):
        self.devices = profile_config.devices
        self.model_names = profile_config.model_names
        self.model_cnt = profile_config.model_cnt
        self.batch_sizes = profile_config.batch_sizes
        self.starts = profile_config.starts
        self.ends = profile_config.ends
        self.mps_list = profile_config.mps_list
        self.dcgm_groups = profile_config.dcgm_groups
        self.log_path = profile_config.path
        self.profile_path = profile_config.profile_path
        self.trace_path = profile_config.trace_path
        self.test_time = profile_config.test_time
        self.seq_len = profile_config.seq_len
        self.rec_pw = profile_config.rec_pw
        self.dirname = os.path.split(os.path.abspath(__file__))[0]
        # The logfile names are set to: client_tag-global_tag-specific_suffix; normally global tag is 
        # an aggregated name of all client tags, but can also be specified specifically
        self.global_tag = profile_config.global_tag
        self.mode = profile_config.mode
        self.scheduler = profile_config.scheduler
        self.pipe_map = {}

    # Currently, within a main run each client's test config mainly remains the same.
    # So save the basic configurations that are barely changed first in self.test_configs for each model client.
    # FIXME: basic configurations actually may be different across runs theoretically, but 
    # current implementation fixes them to be the same (e.g. global_cnt is fixed).
    def get_test_configs(self):
        self.test_configs = {}
        for id, model_name in enumerate(self.model_names):
            if(self.starts[id] == None):
                self.starts[id] = 0
            if(self.ends[id] == None):
                self.ends[id] = self.model_total_modules[model_name]
            test_config = TestConfig(
                batch_size = self.batch_sizes[id],
                start = self.starts[id],
                end = self.ends[id],
                test_time = self.test_time,
                scheduler = self.scheduler,
                trace_path = self.trace_path[id],
                global_cnt = len(self.model_names),
                inter_flag = (self.starts[id] != 0) or (self.ends[id] != self.model_total_modules[model_name])
            )
            if(model_name == "llm"):
                test_config.seq_len = self.seq_len
            self.test_configs[model_name] = test_config
    
    # Get the total_modules of all model clients and save to self.model_total_modules;
    def get_client_total_modules(self):
        self.model_total_modules = {}
        for model_name in self.model_names:
            self.pipe_map[model_name].send("get_total_modules")
            self.model_total_modules[model_name] = self.pipe_map[model_name].recv()
    
    # With the test_configs of all model clients, get their self tags and save to self.self_tags;
    # Self tag of the llm client is also saved to all self.test_configs
    def get_client_self_tags(self):
        self.self_tags = {}
        for model_name in self.model_names:
            self.pipe_map[model_name].send("self_tag")
            self.pipe_map[model_name].send(self.test_configs[model_name])
            self.self_tags[model_name] = self.pipe_map[model_name].recv()
        if("llm" in self.model_names):
            for model_name in self.model_names:
                self.test_configs[model_name].llm_tag = self.self_tags["llm"]
    
    # With the test_configs of all model clients and the global tag, get their full tags and save to self.full_tags;
    def get_client_full_tags(self):
        self.full_tags = {}
        for model_name in self.model_names:
            self.pipe_map[model_name].send("full_tag")
            self.pipe_map[model_name].send(self.test_configs[model_name])
            self.full_tags[model_name] = self.pipe_map[model_name].recv()

    # With the test_configs of all model clients, get global tag as aggregated name if self.global_tag is not defined,
    # else directly use self.global_tag. Save global tags to self.test_configs
    def get_global_tag(self):
        if(not self.global_tag):
            tags = []
            for model_name in self.model_names:
                tags.append(self.self_tags[model_name])
            self.global_tag = '+'.join(tags)
            if(self.scheduler):
                self.global_tag = "-".join([self.global_tag, self.scheduler])
            self.global_tag += f'-d{len(self.devices)}'
        for model_name in self.model_names:
            self.test_configs[model_name].global_tag = self.global_tag
        return self.global_tag

    # Setup actions related to tags: get & save model client self tags, the global tag, and client full tags
    def setup_tags(self):
        self.get_client_self_tags()
        self.get_global_tag()
        self.get_client_full_tags()

    # Set up environment for a test run, include actions:
    # 1. Basic barrier creation
    # 2. Start clients
    # 3. Get the total modules of each models
    # 3. Save the basic test configurations
    # 4. Basic setups related to tags ↑
    def setup(self):
        global model_clients, profile_dict, profile_group_dict
        self.barrier = mp.Barrier(self.model_cnt + 1)
        self.double_barrier = mp.Barrier(2)
        for id, model_name in enumerate(self.model_names):
            pipe_parent, pipe_child = mp.Pipe()
            client = model_clients[model_name](
                self.devices,
                self.mps_list[id],
                pipe_child,
                (self.double_barrier, self.barrier),
                self.log_path,
                self.profile_path
            )
            client.start()
            self.pipe_map[model_name] = pipe_parent
        self.barrier.wait()
        self.get_client_total_modules()
        self.get_test_configs()
        self.setup_tags()

    # PROFILE MODE: Perform profiling on model clients one by one. 
    def profile(self):
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send("profile")
            self.double_barrier.wait()
        print("finished_profiling")

    # TEST MODE: Normal executions of each models by themselves
    # If scheduler is set & llm client is included, the llm client executes on its own but other models 
    # execute in adaptive mode.
    def test(self):
        print("start test:", self.global_tag)

        # Even when dcgm is not profiled, a dcgm_group -1 will be inplace.
        for dcgm_id, dcgm_group in enumerate(self.dcgm_groups):
            for id, model_name in enumerate(self.model_names):
                recording_action = RecordingAction(
                    record_statistics = True if dcgm_id == 0 else False,
                    record_dcgm = True if id == 0 else False,
                    record_pw = self.rec_pw if (dcgm_id == 0 and id == 0) else False
                )
                self.test_configs[model_name].dcgm_group = dcgm_group
                self.test_configs[model_name].recording_action = recording_action

                self.pipe_map[model_name].send("test")
                self.pipe_map[model_name].send(self.test_configs[model_name])
            # Wait for warmup
            self.barrier.wait()
            # Wait for actual test
            self.barrier.wait()
            print("round one end")

    # Shutdown all model clients
    def shutdown(self):
        for model_name in self.model_names:
            self.pipe_map[model_name].send("stop")
        self.barrier.wait()
    
    # Get a model client's log file path with specific filename 
    # Log file path: self.log_path/client_tag-filename
    def get_model_client_log_filepath(self, model_name, filename):
        return os.path.join(self.log_path, "-".join([self.full_tags[model_name], filename]))
    
    def get_log_filepath(self, filename):
        return os.path.join(self.log_path, "-".join([self.global_tag, filename]))

    """
        Collect dcgm data & power data & a full summary of the previous test run.
        Return passed_flag indicating 99% SLO attainment of the LLM service, along with total and passed requests;
        (True, 0, 0) if LLM service non-existing. 
        TODO: current included dcgm data is only the last profile test run instead of all set dcgm profile groups.
        TODO: haven't checked plot fucntion
    """
    def collect_data(self):
        dcgm_data = None
        power_data = None
        passed_flag = True
        total = 0
        passed = 0

        log_filepath = os.path.join(self.log_path, self.global_tag + '-final_summary.log')
        print("Writing final summary to:", log_filepath)
        with open(log_filepath, 'w+') as summary_file:
            for idx, model_name in enumerate(self.model_names):
                self.pipe_map[model_name].send("collect")
                if(idx == 0):
                    self.pipe_map[model_name].send("dcgm")
                    self.pipe_map[model_name].send(self.dcgm_groups)
                    dcgm_data = self.pipe_map[model_name].recv()
                    if(self.rec_pw):
                        self.pipe_map[model_name].send("power")
                        power_data = self.pipe_map[model_name].recv()
                self.pipe_map[model_name].send("stats")
                print("==== Profile:", model_name)
                if(model_name == "llm"):
                    passed_flag, completed, passed, total, slo_info = self.pipe_map[model_name].recv()
                    percentage = passed / total if total else -1
                    print("* Passed:", passed_flag, "Total Completed", completed, "Met", passed, "Total Submitted", total, "Percentage", percentage, "SLO_info", slo_info)
                    summary_file.write(f'{model_name} {passed_flag} {completed}/{passed}/{total} {percentage} {slo_info}\n')
                else:
                    absolute_throughput = self.pipe_map[model_name].recv()
                    print("* Absolute throughput:", absolute_throughput)
                    summary_file.write(model_name + " " + str(absolute_throughput) + "\n")
                self.pipe_map[model_name].send("stop")
        
            if(dcgm_data is not None):
                summary_file.write("============================\n")
                print(f"==== DCGM\n{dcgm_data}")
                for key, value in dcgm_data.items():
                    summary_file.write(key + " " + str(value) + "\n")
            if(power_data is not None):
                summary_file.write("============================\n")
                energy, power, spent_time = power_data
                power_summary_str = f'{energy}(kJ) {power}(W) {spent_time}(s)'
                print(f"==== Power\n{power_summary_str}")
                summary_file.write(f'{power_summary_str}\n')

        if("llm" in self.model_names and len(self.model_names) >= 2):
            submitted_intensity = []
            with open(self.get_model_client_log_filepath("llm", 'avg_submitted_intensity.log'), 'r') as input_file:
                lines = input_file.readlines()
                for line in lines:
                    submitted_intensity.append(tuple(map(float, line.strip('\n').split(' '))))
            submitted_intensity = TimeBinRecord.get_summed_time_bin_record(submitted_intensity).append_timestamps()
            submitted_intensity = TraceAxisData(TracePlotData('Number of submitted reqs', submitted_intensity), y_tag = ('Intensity', ''))
            for model_name in self.model_names:
                if(model_name == "llm"):
                    continue
                throughputs = []
                with open(self.get_model_client_log_filepath(model_name, 'flow.log'), 'r') as input_file:
                    lines = input_file.readlines()
                    for line in lines:
                        parts = line.split()
                        throughputs.append([float(parts[0]), int(parts[1])])  
                throughputs = TimeBinRecord.get_summed_time_bin_record(throughputs, bin_size = 5).append_timestamps()
                side_throughput = TraceAxisData(TracePlotData('Throughput of offline workload', throughputs), y_tag = ('Throughput', ''))
                plot_trace_graph(self.get_log_filepath('submitted_offline_throughput.png'), submitted_intensity, side_throughput)
        return passed_flag, passed, total 




