"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
from torchsummary import summary
import torch
import time
from profiling.models.model_worker import ModelWorker
from profiling.models.split.unet import UNet3D

class UNet3DWorker(ModelWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__("unet3d", device_id, mps, pipe, barrier, log_path, profile_path)
        self.supported_batchsize = [x for x in range(1,17)]
        self.alpha = 1

    def init_start(self):
        self.model = UNet3D(in_channels=3, num_classes=1).cuda(self.device).eval()
        self.random_input = {
            batch_size: torch.rand(batch_size, 3, 16, 128, 128).cuda(self.device)
            for batch_size in self.supported_batchsize
        }
    
    def common_start(self):
        self.total_module = -1