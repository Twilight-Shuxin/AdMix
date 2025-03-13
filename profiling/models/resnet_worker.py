from profiling.models.split.resnet import resnet50
from profiling.models.model_worker import ModelWorker
import torch
import os

class ResnetWorker(ModelWorker):
    def __init__(self, mps, pipe, barrier, log_path):
        super().__init__("resnet50", mps, pipe, barrier, log_path)
    
    def init_start(self):
        self.model = resnet50().cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 224, 224).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
