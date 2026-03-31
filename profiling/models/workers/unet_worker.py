"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

import torch
from profiling.models.workers.split.unet import UNet3D
from profiling.models.workers.model_worker import ModelWorker

class UNet3DWorker(ModelWorker):
    def __init__(self, model_name, devices, supported_batchsize, pipe, worker_barrier, single_worker_barrier, log_path, profile_path):
        super().__init__(model_name, devices, pipe, worker_barrier, single_worker_barrier, log_path, profile_path)
        self.supported_batchsize = supported_batchsize

    def init_start(self):
        self.model = UNet3D(in_channels=3, num_classes=1).cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 16, 128, 128).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
        self.profile_pw_bs = 16