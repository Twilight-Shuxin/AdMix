from profiling.models.model_client import ModelClient
from profiling.models.workers import UNet3DWorker

class UNet3DClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path):
        super().__init__("unet3d", devices, mps, pipe, barrier, log_path, profile_path)
        self.alpha = 1
    
    def init_start(self):
        self.model_worker = UNet3DWorker