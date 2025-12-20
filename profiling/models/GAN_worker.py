from profiling.models.split.gan import GAN
from profiling.models.model_worker import ModelWorker
import torch
import os
import numpy as np
import torch.nn as nn

class GANWorker(ModelWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("GAN", device_id, mps, pipe, barrier, log_path, profile_path)
        self.supported_batchsize = [x for x in range(1,33)]
    
    def init_start(self):
        self.model = GAN().cuda(self.device).eval()
        # self.random_input = {
        #     batch_size: torch.tensor(np.random.normal(0, 1, (batch_size, self.model.latent_dim)), dtype=torch.float32).to(self.device)
        #     for batch_size in self.supported_batchsize
        # }
        self.random_input = {}
        self.label_emb = nn.Embedding(self.model.n_classes, self.model.n_classes).to(self.device)
        for batch_size in self.supported_batchsize:
            labels = torch.tensor(np.random.randint(0, self.model.n_classes, batch_size),  dtype=torch.int64).to(self.device)
            noise = torch.tensor(np.random.normal(0, 1, (batch_size, self.model.latent_dim)), dtype=torch.float32).to(self.device)
            self.random_input[batch_size] = torch.cat((self.label_emb(labels), noise), -1).to(self.device)

    def common_start(self):
        self.total_module = -1