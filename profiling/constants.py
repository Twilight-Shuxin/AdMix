import os
import torch

profile_group_dict = [
    # ['tensor_active'], #0, A1
    ['sm_active', 'sm_occupancy', 'tensor_active', 'fp64_active'], #0, A1
    ['fp16_active', 'tensor_imma_active', 'tensor_hmma_active'], #1, A2
    ['fp32_active'], #2, A3
    ['dram_active'], #3, B0
    ['pcie_tx_bytes', 'pcie_rx_bytes'], #4, C0
    ['gr_engine_active'], #5, D0
    ['nvlink_tx_bytes', 'nvlink_rx_bytes'], #6,E0
]

sub_module_size_dict = {
    "resnet50": 18,
    "resnet101": 35,
    "resnet152": 52,
    "inception_v3": 14,
    "vgg16": 19,
    "vgg19": 22,
    "bert": 12,
}

profile_dict = {
    'sm_active': 1002,
    'sm_occupancy': 1003,
    'tensor_active': 1004,
    'fp64_active': 1006,
    'fp16_active': 1008,
    'tensor_imma_active': 1013,
    'tensor_hmma_active': 1014,
    'fp32_active': 1007,
    'dram_active': 1005,
    'pcie_tx_bytes': 1009,
    'pcie_rx_bytes': 1010,
    'gr_engine_active': 1001,
    'nvlink_tx_bytes': 1011,
    'nvlink_rx_bytes': 1012,
}

class TestConfig():
    def __init__(self, batch_size, start, end, dcgm_group, seq_len = -1):
        self.batch_size = batch_size
        self.start = start
        self.end = end
        self.dcgm_group = dcgm_group
        self.seq_len = seq_len
        self.nosync = False
    
    def fit(self, model_worker):
        if(self.start == -1):
            self.start = 0
        if(self.end == -1):
            self.end = model_worker.total_module
        self.inter_flag = (self.start != 0) or (self.end != model_worker.total_module)
        if(model_worker.model_name != "bert"):
            model_worker.submodel = torch.nn.Sequential(*model_worker.submodules[self.start:self.end])
    
def stop_dcgm(pids):
    for pid in pids:
        command = "kill -9 {}".format(str(pid))
        os.system(command)