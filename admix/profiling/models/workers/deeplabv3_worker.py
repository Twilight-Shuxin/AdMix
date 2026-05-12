import torch
import numpy as np
from admix.profiling.models.workers.model_worker import ModelWorker

class Deeplabv3Worker(ModelWorker):
    def __init__(self, model_name, devices, supported_batchsize, pipe, worker_barrier, single_worker_barrier, log_path, profile_path):
        super().__init__(model_name, devices, pipe, worker_barrier, single_worker_barrier, log_path, profile_path)
        self.supported_batchsize = supported_batchsize

    def init_start(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.tensor(np.random.normal(0, 1, (batch_size, 3, 350, 350)), dtype=torch.float32).to(self.device)
            for batch_size in self.supported_batchsize
        }
        self.profile_pw_bs = 32
