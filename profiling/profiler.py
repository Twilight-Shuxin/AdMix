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
                self.profile_groups[id], self.seq_len)
            self.pipe_map[model_name].send(testconfig)
            self.global_tag += self.pipe_map[model_name].recv()
        return self.global_tag

    def run(self):
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
        
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send(("test"))
            testconfig = TestConfig(self.model_cnt, self.global_tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                self.profile_groups[id], self.seq_len)
            # testconfig.nosync = True
            testconfig.trace_path = self.trace_path[id]
            self.pipe_map[model_name].send(testconfig)
        
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send(("stop"))




