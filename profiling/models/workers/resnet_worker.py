import torch
from profiling.models.workers.model_worker import ModelWorker
from profiling.models.workers.split.resnet import resnet50

class ResnetWorker(ModelWorker):
    def __init__(self, model_name, devices, supported_batchsize, pipe, worker_barrier, single_worker_barrier, log_path, profile_path):
        super().__init__(model_name, devices, pipe, worker_barrier, single_worker_barrier, log_path, profile_path)
        self.supported_batchsize = supported_batchsize
    
    def init_start(self):
        self.model = resnet50().cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 224, 224).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
        self.profile_pw_bs = 128
