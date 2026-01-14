from profiling.models.model_worker import ModelWorker
import torch
import os
import numpy as np

class Deeplabv3Worker(ModelWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("deeplabv3", device_id, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 1
        self.supported_batchsize = [x for x in range(1,33)]

    def init_start(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.tensor(np.random.normal(0, 1, (batch_size, 3, 350, 350)), dtype=torch.float32).to(self.device)
            for batch_size in self.supported_batchsize
        }
