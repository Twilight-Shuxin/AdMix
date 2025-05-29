from profiling.utils import *
from profiling.constants import *
import torch.multiprocessing as mp
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
        self.trace_path = profile_config.trace_path
        self.seq_len = profile_config.seq_len
        self.test_time = profile_config.test_time
        self.dirname = os.path.split(os.path.abspath(__file__))[0]
        self.global_tag = profile_config.global_tag
        self.pipe_map = {}
    
    def get_global_tag(self):
        if(self.global_tag != None):
            return self.global_tag
        if(self.model_cnt == 1):
            return None
        self.global_tag = ""
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send(("tag"))
            testconfig = TestConfig(self.model_cnt, None, self.batch_sizes[id], self.starts[id], self.ends[id], 
                -1, self.seq_len)
            self.pipe_map[model_name].send(testconfig)
            self.global_tag += self.pipe_map[model_name].recv()
        return self.global_tag

    def setup(self):
        global model_workers, profile_dict, profile_group_dict
        self.barrier = mp.Barrier(self.model_cnt + 1)
        for id, model_name in enumerate(self.model_names):
            pipe_parent, pipe_child = mp.Pipe()
            if(self.mps_list[id] != -1):
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.mps_list[id])
            worker = model_workers[model_name](
                self.device,
                self.mps_list[id],
                pipe_child,
                self.barrier,
                self.log_path
            )
            worker.start()
            self.pipe_map[model_name] = pipe_parent
        self.barrier.wait()
        self.get_global_tag()

    def test(self, serve_flag = False):
        print("start test")
        for prof_id, prof_group in enumerate(self.profile_groups):
            for id, model_name in enumerate(self.model_names):
                self.pipe_map[model_name].send("test" if(model_name == "llm" or serve_flag == False) else "opp_serve")
                if (prof_id == 0):
                    recording_action = RecordingAction.ALL if (id == 0) else RecordingAction.STATISTICS_ONLY
                elif (id == 0):
                    recording_action = RecordingAction.DCGM_ONLY
                else:
                    recording_action = RecordingAction.NO_RECORD
                test_config = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                    prof_group, self.seq_len, test_time = self.test_time, recording_action = recording_action)
                if(serve_flag):
                    test_config.scheduler = "AIMD"
                # test_config.nosync = True
                test_config.trace_path = self.trace_path[id]
                self.pipe_map[model_name].send(test_config)
            self.barrier.wait()
            print("round one end")
    
    def serve(self):
        self.test(True)

    def shutdown(self):
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send("stop")
        self.barrier.wait()
    
    def collect_data(self):
        dcgm_data = None
        tag = self.global_tag
        if(tag == None):
            model_name = next(iter(self.pipe_map))
            test_config = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[0], self.starts[0], self.ends[0], 
                    self.profile_groups, self.seq_len)
            self.pipe_map[model_name].send("tag")
            self.pipe_map[model_name].send(test_config)
            tag = self.pipe_map[model_name].recv()
        log_filepath = os.path.join(self.log_path, tag + '-final_summary.log')
        print(log_filepath)
        with open(log_filepath, 'w+') as summary_file:
            for id, model_name in enumerate(self.model_names):
                self.pipe_map[model_name].send("collect")
                test_config = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                    self.profile_groups, self.seq_len)
                # test_config.nosync = True
                if(id != 0 or self.profile_groups[0] == -1):
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
                    passed_flag, passed, total_fixed, total, slo_info = self.pipe_map[model_name].recv()
                    print("* Passed:", passed_flag, "Met", passed, "Total Fixed", total_fixed, "Total Submitted", total, "SLO_info", slo_info)
                    summary_file.write(model_name + " " + str(passed_flag) + " " + str(passed) + "/" + str(total_fixed) + "/" + str(total) + " " + str(slo_info) + "\n")
            summary_file.write("============================\n")
            if(dcgm_data != None):
                print(dcgm_data)
                for key, value in dcgm_data.items():
                    summary_file.write(key + " " + str(value) + "\n")


    def run(self):
        self.setup()
        self.test()
        self.shutdown()




