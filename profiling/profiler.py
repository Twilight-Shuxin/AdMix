from profiling.utils import *
from profiling.constants import *
import torch.multiprocessing as mp
from plotting.trace_plot import *
from trace.trace_utils import set_burstgpt_trace_config, read_burstgpt_trace_config_json
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
                set_burstgpt_trace_config(trace_config_id)
                trace_config = read_burstgpt_trace_config_json("trace_config.json")
                if("time" in trace_config):
                    self.test_time = trace_config["time"]
        self.seq_len = profile_config.seq_len
        self.dirname = os.path.split(os.path.abspath(__file__))[0]
        self.global_tag = profile_config.global_tag
        self.mode = profile_config.mode
        self.scheduler = profile_config.scheduler
        self.pipe_map = {}
    
    def get_global_tag(self):
        if(self.global_tag != None):
            return self.global_tag
        tags = []
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send(("tag"))
            testconfig = TestConfig(self.model_cnt, None, self.batch_sizes[id], self.starts[id], self.ends[id], 
                None, self.seq_len)
            testconfig.trace_path = self.trace_path[id]
            self.pipe_map[model_name].send(testconfig)
            tags.append(self.pipe_map[model_name].recv())
        self.global_tag = '+'.join(tags)
        if(self.scheduler != None):
            self.global_tag = "-".join([self.global_tag, self.scheduler])
        return self.global_tag

    def setup(self):
        global model_workers, profile_dict, profile_group_dict
        self.barrier = mp.Barrier(self.model_cnt + 1)
        self.single_barrier = mp.Barrier(2)
        for id, model_name in enumerate(self.model_names):
            pipe_parent, pipe_child = mp.Pipe()
            if(self.mps_list[id] != None):
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.mps_list[id])
            worker = model_workers[model_name](
                self.device,
                self.mps_list[id],
                pipe_child,
                (self.single_barrier, self.barrier),
                self.log_path,
                self.profile_path
            )
            worker.start()
            self.pipe_map[model_name] = pipe_parent
        self.barrier.wait()
        self.get_global_tag()

    def profile(self):
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send("profile")
            self.single_barrier.wait()
        print("finished_profiling")

    def test(self, scheduler = None):
        print("start test", self.global_tag)
        for prof_id, prof_group in enumerate(self.profile_groups):
            for id, model_name in enumerate(self.model_names):
                self.pipe_map[model_name].send("test" if(model_name == "llm" or scheduler == None) else "opp_serve")
                if (prof_id == 0):
                    recording_action = RecordingAction.ALL if (id == 0) else RecordingAction.STATISTICS_ONLY
                elif (id == 0):
                    recording_action = RecordingAction.DCGM_ONLY
                else:
                    recording_action = RecordingAction.NO_RECORD
                test_config = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                    prof_group, self.seq_len, test_time = self.test_time, recording_action = recording_action)
                if(scheduler != None):
                    test_config.scheduler = scheduler
                # test_config.nosync = True
                test_config.trace_path = self.trace_path[id]
                self.pipe_map[model_name].send(test_config)
            self.barrier.wait()
            print("round one end")
    
    def serve(self):
        self.test(self.scheduler)
        # self.test("advanced_AIMD")
        # self.test(None)

    def shutdown(self):
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send("stop")
        self.barrier.wait()
    
    def collect_tags(self):
        self.model_worker_tags = {}
        for id, model_name in enumerate(self.model_names):
            test_config = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                    self.profile_groups, self.seq_len)
            test_config.trace_path = self.trace_path[id]
            self.pipe_map[model_name].send("tag")
            self.pipe_map[model_name].send(test_config)
            self.model_worker_tags[model_name] = self.pipe_map[model_name].recv()
    
    def get_model_worker_filename_with_tag(self, model_name, tag, filename):
        tag = "-".join([model_name, self.global_tag]) if ((self.global_tag is not None) and (tag != self.global_tag)) else tag
        return os.path.join(self.log_path, "-".join([tag, filename]))

    # should first collect tags
    def get_model_worker_filename(self, model_name, filename):
        return self.get_model_worker_filename_with_tag(model_name, self.model_worker_tags[model_name], filename)
    
    def collect_data(self):
        dcgm_data = None
        tag = self.global_tag
        self.collect_tags()
        if(tag == None):
            tag = random.choice(list(self.model_worker_tags.values()))
        log_filepath = os.path.join(self.log_path, tag + '-final_summary.log')
        print(log_filepath)
        with open(log_filepath, 'w+') as summary_file:
            for id, model_name in enumerate(self.model_names):
                self.pipe_map[model_name].send("collect")
                test_config = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                    self.profile_groups, self.seq_len)
                # test_config.nosync = True
                if(id != 0 or self.profile_groups[0] == None):
                    test_config.dcgm_group = None
                test_config.trace_path = self.trace_path[id]
                self.pipe_map[model_name].send(test_config)
                if(test_config.dcgm_group != None):
                    dcgm_data = self.pipe_map[model_name].recv()
                print("==== Profile:", model_name)
                if(model_name != "llm"):
                    absolute_throughput, throughput = self.pipe_map[model_name].recv()
                    print("* Absolute throughput:", absolute_throughput, "Throughput", throughput)
                    summary_file.write(model_name + " " + str(absolute_throughput) + " " + str(throughput) + "\n")
                else:
                    passed_flag, completed, passed, total, slo_info = self.pipe_map[model_name].recv()
                    print("* Passed:", passed_flag, "Total Completed", completed, "Met", passed, "Total Submitted", total, "SLO_info", slo_info)
                    summary_file.write(model_name + " " + str(passed_flag) + " " + str(completed) + "/" + str(passed) + "/" + str(total) + " " + str(slo_info) + "\n")
            summary_file.write("============================\n")
            if(dcgm_data != None):
                print(dcgm_data)
                for key, value in dcgm_data.items():
                    summary_file.write(key + " " + str(value) + "\n")

        # In this case global tag must be existing
        if("llm" in self.model_names and len(self.model_names) >= 2):
            intensity_list = []
            with open(self.get_model_worker_filename("llm", 'aggregated_intensity_all.log'), 'r') as input_file:
                lines = input_file.readlines()
                for line in lines:
                    bin_midpoint_data, intensity_data = map(float, line.strip('\n').split(' '))
                    intensity_list.append([bin_midpoint_data, intensity_data]) 
            for id, model_name in enumerate(self.model_names):
                if(model_name == "llm"):
                    continue
                values = []
                with open(self.get_model_worker_filename(model_name, 'flow.log'), 'r') as input_file:
                    lines = input_file.readlines()
                    for line in lines:
                        parts = line.split()  # Split by whitespace
                        values.append([float(parts[0]), int(parts[1])])  
                
                trace_plot_aggregated_intensity_with_analysis_data(intensity_list, values, ('Side Absolute Throughput', ''), 
                    self.get_model_worker_filename(model_name, 'aggregated_intensity_absolute_throughput.png'))


    def run(self):
        self.setup()
        self.test()
        self.shutdown()




