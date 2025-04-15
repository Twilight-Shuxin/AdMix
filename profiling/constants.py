import os
import torch
from enum import Enum
from trace.trace_utils import Trace

class ArgsType(Enum):
    INT = 1
    STRING = 2

GPU_UUID = [
    "GPU-dbbfc53e-396c-4cfb-bce3-88dc1c760e98",
    "GPU-52d1d800-a5dc-f06d-1e17-7778d35b591b"
]

def device_init(device_id):
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_UUID[device_id]
    os.environ["CUDA_MPS_PIPE_DIRECTORY"]="/tmp/nvidia-mps-" + GPU_UUID[device_id]
    os.environ["CUDA_MPS_LOG_DIRECTORY"]="/tmp/nvidia-mps-" + GPU_UUID[device_id]


def background(cmd, log):
    logger.info(" ".join(cmd))
    log = gmanager_home + "/" + log
    open(log, 'w').close() 
    return subprocess.Popen(cmd,
            stdin=subprocess.DEVNULL,
            stdout=open(log, 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: (signal.signal(signal.SIGHUP, signal.SIG_IGN)),
            cwd=dir)

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

def get_comma_list(strlist, argstype, expand = None, default = None):
    results = strlist.strip("\n").split(",")
    if(argstype == ArgsType.INT):
        results = [int(result) if result != '' else default for result in results]
    else:
        results = [result if result != '' else default for result in results]
    if(expand is not None):
        if(len(results) != expand):
            if(default == None):
                default = results[0]
            for i in range(len(results), expand):
                results.append(default)
    return results

def get_comma_list_or_single(strlist, argstype, expand = None, default = None):
    comma_list = get_comma_list(strlist, argstype, expand, default)
    if(len(comma_list) == 1): 
        return comma_list[0]
    return comma_list

def list_to_string(data, delimiter = " "):
    data_string = ""
    data = [str(x) for x in data]
    return delimiter.join(data)

class TestConfig():
    def __init__(self, global_cnt, tag, batch_size, start, end, dcgm_group, seq_len = None):
        self.global_cnt = global_cnt
        self.tag = tag
        self.batch_size = batch_size
        self.start = start
        self.end = end
        self.dcgm_group = dcgm_group
        self.seq_len = seq_len
        self.nosync = False
        self.trace_path = None
    
    def get_tag(self, model_worker):
        self_tag = model_worker.get_tag(self)
        self.tag = "-".join([self_tag, self.tag]) if self.tag is not None else self_tag
    
    def fit(self, model_worker):
        if(self.start == -1):
            self.start = 0
        if(self.end == -1):
            self.end = model_worker.total_module
        self.inter_flag = (self.start != 0) or (self.end != model_worker.total_module)
        if(model_worker.model_name != "bert" and self.inter_flag):
            model_worker.submodel = torch.nn.Sequential(*model_worker.submodules[self.start:self.end])
        self.get_tag(model_worker)
        if(self.trace_path != None):
            model_worker.get_trace(self.trace_path)
    
def stop_dcgm(pids):
    for pid in pids:
        command = "kill -9 {}".format(str(pid))
        os.system(command)