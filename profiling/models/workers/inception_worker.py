import torch
from profiling.models.workers.split.inception import inception_v3
from profiling.models.workers.model_worker import ModelWorker

class InceptionWorker(ModelWorker):
    def __init__(self, pipe, worker_barrier, single_worker_barrier):
        super().__init__(pipe, worker_barrier, single_worker_barrier)
        self.supported_batchsize = [x for x in range(1,33)]

    def init_start(self):
        self.model = inception_v3().cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 299, 299).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
