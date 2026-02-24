import torch
import numpy as np
from profiling.models.workers.model_worker import ModelWorker

class Deeplabv3Worker(ModelWorker):
    def __init__(self, pipe, worker_barrier, single_worker_barrier):
        super().__init__(pipe, worker_barrier, single_worker_barrier)
        self.supported_batchsize = [x for x in range(1,33)]

    def init_start(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.tensor(np.random.normal(0, 1, (batch_size, 3, 350, 350)), dtype=torch.float32).to(self.device)
            for batch_size in self.supported_batchsize
        }
