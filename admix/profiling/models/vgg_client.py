from admix.profiling.models.model_client import ModelClient
from admix.profiling.models.workers import VggWorker

class VggClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path, llm_model=None):
        super().__init__("vgg19", devices, mps, pipe, barrier, log_path, profile_path, llm_model=llm_model)
        self.supported_batchsize = [x for x in range(1,129)]
        self.alpha = 8
    
    def init_start(self):
        self.model_worker = VggWorker