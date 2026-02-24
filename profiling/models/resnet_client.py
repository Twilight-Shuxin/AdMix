from profiling.models.model_client import ModelClient
from profiling.models.workers import ResnetWorker

class ResnetClient(ModelClient):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("resnet50", device_id, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 8
    
    def init_start(self):
        self.model_worker = ResnetWorker