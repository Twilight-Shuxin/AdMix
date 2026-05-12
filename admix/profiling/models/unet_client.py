from admix.profiling.models.model_client import ModelClient
from admix.profiling.models.workers import UNet3DWorker

class UNet3DClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path, llm_model=None):
        super().__init__("unet3d", devices, mps, pipe, barrier, log_path, profile_path, llm_model=llm_model)
        self.supported_batchsize = [x for x in range(1,17)]
        self.alpha = 1
    
    def init_start(self):
        self.model_worker = UNet3DWorker