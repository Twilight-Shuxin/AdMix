from profiling.models.model_client import ModelClient
from profiling.models.workers import InceptionWorker

class InceptionClient(ModelClient):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("inception_v3", device_id, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 1
    
    def init_start(self):
        self.model_worker = InceptionWorker