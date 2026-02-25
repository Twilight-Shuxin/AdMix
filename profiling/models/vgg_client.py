from profiling.models.model_client import ModelClient
from profiling.models.workers import VggWorker

class VggClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path):
        super().__init__("vgg19", devices, mps, pipe, barrier, log_path, profile_path)
        self.supported_batchsize = [x for x in range(1,129)]
        self.alpha = 8
    
    def init_start(self):
        self.model_worker = VggWorker