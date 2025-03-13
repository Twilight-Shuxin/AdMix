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
        self.pipe_map = {}

    def run(self):
        global model_worker, profile_dict, profile_group_dict
        self.barrier = mp.Barrier(self.model_cnt + 1)
        for id, model_name in enumerate(self.model_names):
            pipe_parent, pipe_child = mp.Pipe()
            if(self.mps_list[id] != -1):
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.mps_list[id])
            worker = model_worker[model_name](
                self.mps_list[id],
                pipe_child,
                self.barrier,
                self.log_path
            )
            worker.start()
            self.pipe_map[model_name] = pipe_parent
        self.barrier.wait()
        tag = "".join(self.model_names)
        if(self.model_cnt == 1):
            tag = None
        elif(self.mps_list[0] != -1):
            tag = ""
            for id, model_name in enumerate(self.model_names):
                if(model_name != "llm"):
                    tag += model_name + "(" + str(self.batch_sizes[id])  + "_" + str(self.mps_list[id]) + ")"
                else: 
                    tag += model_name + "(" + str(self.mps_list[id]) + ")"
        for id, model_name in enumerate(self.model_names):
            self.pipe_map[model_name].send(("test"))
            testconfig = TestConfig(self.model_cnt, tag, self.batch_sizes[id], self.starts[id], self.ends[id], 
                self.profile_groups[id], self.seq_len)
            # testconfig.nosync = True
            testconfig.trace_path = self.trace_path[id]
            self.pipe_map[model_name].send(testconfig)




