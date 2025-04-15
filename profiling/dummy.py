from llm.llm_worker_test import *
from profiling.models.resnet_worker import ResnetWorker
from profiling.constants import *
import torch.multiprocessing as mp
import subprocess 
import time
import os

class Dummy():
    def __init__(self, profile_config):
        self.profile_config = profile_config
        device_init(profile_config.device)

    def run(self):
        self.barrier = mp.Barrier(2)
        id = 0
        pipe_parent, pipe_child = mp.Pipe()
        if(self.profile_config.mps_list[id] != -1):
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.profile_config.mps_list[id])
        worker = LLMWorkerTest(
            self.profile_config.device,
            self.profile_config.mps_list[id],
            pipe_child,
            self.barrier,
            self.profile_config.path
        )
        worker.start()
        self.barrier.wait()




