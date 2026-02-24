from profiling.models.model_client import ModelClient
from profiling.models.workers import GANWorker

class GANClient(ModelClient):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("GAN", device_id, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 1
    
    def init_start(self):
        self.model_worker = GANWorker