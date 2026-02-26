import torch
from profiling.models.workers.model_worker import ModelWorker
from profiling.models.workers.split.inception import inception_v3

class InceptionWorker(ModelWorker):
    def __init__(self, supported_batchsize, pipe, worker_barrier, single_worker_barrier, log_path, profile_path):
        super().__init__(pipe, worker_barrier, single_worker_barrier, log_path, profile_path)
        self.supported_batchsize = supported_batchsize

    def init_start(self):
        self.model = inception_v3().cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 299, 299).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
