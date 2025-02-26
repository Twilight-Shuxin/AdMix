from profiling.models.split.resnet import resnet50
from profiling.models.model_worker import ModelWorker
import torch

class ResnetWorker(ModelWorker):
    def __init__(self, device, pipe, log_path):
        super().__init__("resnet50", device, pipe, log_path)
    
    def init_start(self):
        self.model = resnet50().cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 224, 224).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
