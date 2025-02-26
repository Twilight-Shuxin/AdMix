from profiling.utils import *
from profiling.constants import *
import torch.multiprocessing as mp
import subprocess 
import time
import os

class ProfileServer():
    def __init__(self, profile_config):
        self.device = profile_config.device
        self.model_name = profile_config.model_name
        self.batch_size = profile_config.batch_size
        self.start = profile_config.start
        self.end = profile_config.end
        self.profile_group = profile_config.profile_group
        self.log_path = profile_config.path
        self.dirname = os.path.split(os.path.abspath(__file__))[0]

    def run(self):
        pipe_parent, pipe_child = mp.Pipe()
        global model_worker, profile_dict, profile_group_dict
        model_worker = model_worker[self.model_name](
            self.device, 
            pipe_child,
            self.log_path
        )
        model_worker.start()
        pipe_parent.send(("test"))
        testconfig = TestConfig(self.batch_size, self.start, self.end, self.profile_group)
        # testconfig.nosync = True
        pipe_parent.send(testconfig)




