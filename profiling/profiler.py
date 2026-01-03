from profiling.utils import *
from profiling.constants import *
import torch.multiprocessing as mp
from plotting.trace_plot import *
from trace.trace_utils import set_trace_config, read_trace_config_json
import random
import subprocess 
import time
import os

class ProfileServer():
    def __init__(self, profile_config):
        self.device = profile_config.device
        device_init(self.device)
        self.model_names = profile_config.model_names
        self.model_cnt = profile_config.model_cnt
        self.batch_sizes = profile_config.batch_sizes
        self.starts = profile_config.starts
        self.ends = profile_config.ends
        self.mps_list = profile_config.mps_list
        self.profile_groups = profile_config.profile_groups
        self.log_path = profile_config.path
        self.profile_path = profile_config.profile_path
        self.trace_path = profile_config.trace_path
        self.test_time = profile_config.test_time
        for id, trace in enumerate(self.trace_path):
            if(trace != None and "[" in trace):
                self.trace_path[id], trace_config_id = trace.split("[")
                trace_config_id = int(trace_config_id.strip("]"))
                set_trace_config(trace_config_id)
                trace_config = read_trace_config_json("trace_config.json")
                # Prioritize test time in cmd args
                if(self.test_time is None and "time" in trace_config):
                    self.test_time = trace_config["time"]
        self.seq_len = profile_config.seq_len
        self.dirname = os.path.split(os.path.abspath(__file__))[0]
        # The logfile names are set to: worker_tag-global_tag-specific_suffix; normally global tag is 
        # an aggregated name of all worker tags, but can also be specified specifically
        self.global_tag = profile_config.global_tag
        self.mode = profile_config.mode
        self.scheduler = profile_config.scheduler
        self.pipe_map = {}

    # Currently, within a main run each worker's test config mainly remains the same.
    # So save the basic configurations that are barely changed first in self.test_configs for each model worker.
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
                dcgm_group = self.dcgm_group,
                test_time = self.test_time,
                recording_action = recording_action,
                scheduler = self.scheduler,
                trace_path = self.trace_path[id],
                inter_flag = (self.starts[id] != 0) or (self.ends[id] != self.model_total_modules[model_name])
            )
            if(model_name == "llm"):
                test_config.seq_len = self.seq_len
            self.test_configs[model_name] = test_config
    
    # Get the total_modules of all model workers and save to self.model_total_modules;
    def get_worker_total_modules(self):
        self.model_total_modules = {}
        for model_name in self.model_names:
            self.pipe_map[model_name].send(("get_total_modules"))
            self.model_total_modules[model_name] = self.pipe_map[model_name].recv()
    
    # With the test_configs of all model workers, get their self tags and save to self.self_tags;
    # Self tag of the llm worker is also saved to all self.test_configs
    def get_worker_self_tags(self):
        self.self_tags = {}
        for model_name in self.model_names:
            self.pipe_map[model_name].send(("self_tag"))
            self.pipe_map[model_name].send(self.test_configs[model_name])
            self.self_tags[model_name] = self.pipe_map[model_name].recv()
        if("llm" in self.model_names):
            for model_name in self.model_names:
                self.test_configs[model_name].llm_tag = self.self_tags["llm"]
    
    # With the test_configs of all model workers and the global tag, get their full tags and save to self.full_tags;
    def get_worker_full_tags(self):
        self.full_tags = {}
        for model_name in self.model_names:
            self.pipe_map[model_name].send(("full_tag"))
            self.pipe_map[model_name].send(self.test_configs[model_name])
            self.full_tags[model_name] = self.pipe_map[model_name].recv()

    # With the test_configs of all model workers, get global tag as aggregated name if self.global_tag is not defined,
    # else directly use self.global_tag. Save global tags to self.test_configs
    def get_global_tag(self):
        if(self.global_tag == None):
            tags = []
            for model_name in self.model_names:
                tags.append(self.self_tags[model_name])
            self.global_tag = '+'.join(tags)
            if(self.scheduler != None):
                self.global_tag = "-".join([self.global_tag, self.scheduler])
        for model_name in self.model_names:
            self.test_configs[model_name].global_tag = self.global_tag
        return self.global_tag

    # Setup actions related to tags: get & save model worker self tags, the global tag, and worker full tags
    def setup_tags(self):
        self.get_worker_self_tags()
        self.get_global_tag()
        self.get_worker_full_tags()

    # Set up environment for a test run, include actions:
    # 1. Basic barrier creation
    # 2. Start workers
    # 3. Get the total modules of each models
    # 3. Save the basic test configurations
    # 4. Basic setups related to tags ↑
    def setup(self):
        global model_workers, profile_dict, profile_group_dict
        self.barrier = mp.Barrier(self.model_cnt + 1)
        self.double_barrier = mp.Barrier(2)
        for id, model_name in enumerate(self.model_names):
            pipe_parent, pipe_child = mp.Pipe()
            if(self.mps_list[id] != None):
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.mps_list[id])
            worker = model_workers[model_name](
                self.device,
                self.mps_list[id],
                pipe_child,
                (self.double_barrier, self.barrier),
                self.log_path,
                self.profile_path
            )
            worker.start()
            self.pipe_map[model_name] = pipe_parent
        self.barrier.wait()
        self.get_worker_total_modules()
        self.get_test_configs()
        self.setup_tags()

    # PROFILE MODE: Perform profiling on model workers one by one. 
    def profile(self):
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send("profile")
            self.double_barrier.wait()
        print("finished_profiling")

    # TEST MODE: Normal executions of each models by themselves
    # If scheduler is set & llm worker is included, the llm worker executes on its own but other models 
    # execute in adaptive mode.
    def test(self):
        print("start test:", self.global_tag)
        for model_names in self.model_names:
            self.test_configs[model_name].global_cnt = len(self.model_names)

        for prof_id, prof_group in enumerate(self.profile_groups):
            for id, model_name in enumerate(self.model_names):
                if (prof_id == 0):
                    recording_action = RecordingAction.ALL if (id == 0) else RecordingAction.STATISTICS_ONLY
                elif (id == 0):
                    recording_action = RecordingAction.DCGM_ONLY
                else:
                    recording_action = RecordingAction.NO_RECORD
                test_configs[model_name].dcgm_group = prof_group
                test_configs[model_name].set_recording_action(recording_action)

                self.pipe_map[model_name].send("test" if(model_name == "llm" or self.scheduler == None) else "adaptive_serve")
                self.pipe_map[model_name].send(self.test_configs[model_name])
            self.barrier.wait()
            print("round one end")

    # Shutdown all model workers
    def shutdown(self):
        for model_name in self.model_names:
            self.pipe_map[model_name].send("stop")
        self.barrier.wait()
    
    # Get a model worker's log file path with specific filename 
    # Log file path: self.log_path/worker_tag-filename
    def get_model_worker_log_filepath(self, model_name, filename):
        return os.path.join(self.log_path, "-".join([self.full_tags[model_name], filename]))

    """
        Collect dcgm data & a full summary of the previous test run.
        Return passed_flag indicating 99% SLO attainment of the LLM service, along with total and passed requests;
        (True, 0, 0) if LLM service non-existing. 
        TODO: current included dcgm data is only the last profile test run instead of all set dcgm profile groups.
        TODO: haven't checked plot fucntion
    """
    def collect_data(self):
        passed_flag = True
        total = 0
        passed = 0

        log_filepath = os.path.join(self.log_path, self.global_tag + '-final_summary.log')
        print("Writing final summary to:", log_filepath)
        with open(log_filepath, 'w+') as summary_file:
            for model_name in self.model_names:
                self.pipe_map[model_name].send("collect")
                if(self.test_configs[model_name].dcgm_group != None):
                    dcgm_data = self.pipe_map[model_name].recv()
                print("==== Profile:", model_name)
                if(model_name == "llm"):
                    passed_flag, completed, passed, total, slo_info = self.pipe_map[model_name].recv()
                    print("* Passed:", passed_flag, "Total Completed", completed, "Met", passed, "Total Submitted", total, "Percentage", passed / total, "SLO_info", slo_info)
                    summary_file.write(model_name + " " + str(passed_flag) + " " + str(completed) + "/" + str(passed) + "/" + str(total) + " " + str(passed / total) + " " + str(slo_info) + "\n")
                else:
                    absolute_throughput, throughput = self.pipe_map[model_name].recv()
                    print("* Absolute throughput:", absolute_throughput, "Throughput", throughput)
                    summary_file.write(model_name + " " + str(absolute_throughput) + " " + str(throughput) + "\n")
            
            summary_file.write("============================\n")
            if(dcgm_data != None):
                print(dcgm_data)
                for key, value in dcgm_data.items():
                    summary_file.write(key + " " + str(value) + "\n")

        if("llm" in self.model_names and len(self.model_names) >= 2):
            intensity_list = []
            with open(self.get_model_worker_log_filepath("llm", 'aggregated_intensity_all.log'), 'r') as input_file:
                lines = input_file.readlines()
                for line in lines:
                    bin_midpoint_data, intensity_data = map(float, line.strip('\n').split(' '))
                    intensity_list.append([bin_midpoint_data, intensity_data]) 
            for model_name in self.model_names:
                if(model_name == "llm"):
                    continue
                values = []
                with open(self.get_model_worker_log_filepath(model_name, 'flow.log'), 'r') as input_file:
                    lines = input_file.readlines()
                    for line in lines:
                        parts = line.split()
                        values.append([float(parts[0]), int(parts[1])])  
                trace_plot_aggregated_intensity_with_analysis_data(intensity_list, values, ('Side Absolute Throughput', ''), 
                    self.get_model_worker_log_filepath(model_name, 'aggregated_intensity_absolute_throughput.png'))
        return passed_flag, passed, total 


    def run(self):
        self.setup()
        self.test()
        self.shutdown()




