from profiling.models.split.inception import inception_v3
from profiling.models.model_worker import ModelWorker
import torch
import os

class InceptionWorker(ModelWorker):
    def __init__(self, mps, pipe, barrier, log_path):
        super().__init__("inception_v3", mps, pipe, barrier, log_path)

    def init_start(self):

        self.model = inception_v3().cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 299, 299).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
