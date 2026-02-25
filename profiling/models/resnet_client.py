from profiling.models.workers import ResnetWorker
from profiling.models.model_client import ModelClient

class ResnetClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path):
        super().__init__("resnet50", devices, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 8
    
    def init_start(self):
        self.model_worker = ResnetWorker