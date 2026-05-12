from admix.profiling.models.workers import GANWorker
from admix.profiling.models.model_client import ModelClient

class GANClient(ModelClient):
    def __init__(self, devices, mps, pipe, barrier, log_path, profile_path, llm_model=None):
        super().__init__("GAN", devices, mps, pipe, barrier, log_path, profile_path, llm_model=llm_model)
        self.supported_batchsize = [x for x in range(1,129)]
        self.alpha = 1
    
    def init_start(self):
        self.model_worker = GANWorker