from profiling.models.model_client import ModelClient
from profiling.models.workers import Deeplabv3Worker

class Deeplabv3Client(ModelClient):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("deeplabv3", device_id, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 1
    
    def init_start(self):
        self.model_worker = Deeplabv3Worker