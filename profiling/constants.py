import os
import torch
from enum import Enum
from trace.trace_utils import Trace
import numpy as np
import heapq
from collections import deque
import time
import heapq
from collections import Counter

EPS = 1e-6

class ArgsType(Enum):
    INT = 1
    STRING = 2

class RecordingAction(Enum):
    NO_RECORD = 0
    DCGM_ONLY = 1
    STATISTICS_ONLY = 2
    ALL = 3

"""
    Runtime Mode of profiler.
"""
class RuntimeMode(Enum):
    TEST = "TEST" # Original mode with all free functionalities
    PROFILE = "PROFILE" # Profile single model worker
    SERVE = "SERVE" # Serve LLM with other models utilizing scheudler (None:no specific scheduler)

RECORDING_ACTION_MAP = {
    "dcgm": [RecordingAction.DCGM_ONLY, RecordingAction.ALL],
    "statistics": [RecordingAction.STATISTICS_ONLY, RecordingAction.ALL]
}

GPU_UUID = [
    "GPU-dbbfc53e-396c-4cfb-bce3-88dc1c760e98",
    "GPU-52d1d800-a5dc-f06d-1e17-7778d35b591b"
]

def device_init(device_id):
    UUID = GPU_UUID[int(device_id)] if device_id.isdigit() else device_id
    os.environ["CUDA_VISIBLE_DEVICES"]=UUID
    os.environ["CUDA_MPS_PIPE_DIRECTORY"]="/tmp/nvidia-mps-" + UUID
    os.environ["CUDA_MPS_LOG_DIRECTORY"]="/tmp/nvidia-mps-" + UUID


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

"""
    Given a string list 'x1, x2, x3' (ending \n will be stripped);
    Change it into list [x1, x2, x3];
    If argstype is int, the list will be a int list;
    If given expend (int), it requires the list to be expanded into [x1, x2, ..., x_{expand}]
    If default is given, expand list with default; else expand with x1. 
"""
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

"""
    Same functionality as get_comma_list↑, only with 'x1' return x1 instead of [x1]
"""
def get_comma_list_or_single(strlist, argstype, expand = None, default = None):
    comma_list = get_comma_list(strlist, argstype, expand, default)
    if(len(comma_list) == 1): 
        return comma_list[0]
    return comma_list

"""
    Given a record_list that a_x is the record for time interval [t_x, t_{x + 1}] (that is [bin_size * x, bin_size * (x + 1)]),
    expand it into [(bin_size * x, a_x)]
"""
def bin_records_append_timestamps(record_list, bin_size):
    return [[i * bin_size, value] for i, value in enumerate(record_list)]


"""
    Given a list / tuple data, change it into a string with delimiter.
    [1,2] (1,2) -> "1 2" (delimiter = " ")
    If data is already string, then no change is done
"""
def list_to_string(data, delimiter = " "):
    data_string = ""
    if (isinstance(data, list) or isinstance(data, tuple)):
        data = [str(x) for x in data]
    else:
        data = [str(data)]
    return delimiter.join(data)

"""
    A class used to pack all configs related to getting a input tensor.
    batch size: /
    no sync: True / False, if set to true always get a input tensor of batch size 1 
            (input data will be manually submitted up to batch size times at a time without synchronizing the GPU)
    inter_flag: True / False, if set to true means get a partial input
    start: id, indicating which partial result to fetch. get the result generated after executing the (id - 1) operators
         (operator count start from 0)
"""
class InputConfig():
    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else None
        self.nosync = kwargs["nosync"] if "nosync" in kwargs else None
        self.inter_flag = kwargs["inter_flag"] if "inter_flag" in kwargs else None
        self.start = kwargs["start"] if "start" in kwargs else None
        self.seqlen = kwargs["seq_len"] if "seq_len" in kwargs else None

"""
    A class used to pack all configs related to a running work for the worker to perform.
"""
class TestConfig():
    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 1
        self.start = kwargs["start"] if "start" in kwargs else None
        self.end = kwargs["end"] if "end" in kwargs else None
        self.global_tag = kwargs["global_tag"] if "global_tag" in kwargs else None
        self.dcgm_group = kwargs["dcgm_group"] if "dcgm_group" in kwargs else -1
        self.seq_len = kwargs["seq_len"] if "seq_len" in kwargs else None
        self.nosync = kwargs["nosync"] if "nosync" in kwargs else False
        self.trace_path = kwargs["trace_path"] if "trace_path" in kwargs else None
        self.set_recording_action(kwargs["recording_action"] if "recording_action" in kwargs else RecordingAction.NO_RECORD)
        self.scheduler = kwargs["scheduler"] if "scheduler" in kwargs else None
        self.global_cnt = kwargs["global_cnt"] if "global_cnt" in kwargs else 1
        self.inter_flag = kwargs["inter_flag"] if "inter_flag" in kwargs else False
        self.test_time = kwargs["test_time"] if "test_time" in kwargs else None
    
    def set_recording_action(self, recording_action):
        self.recording_action = recording_action
        self.record_statistics = (self.recording_action in RECORDING_ACTION_MAP["statistics"])
        self.record_dcgm = (self.recording_action in RECORDING_ACTION_MAP["dcgm"])

"""
    A class used to hold runtime data generated in a test run performed by a model worker.
"""
class RuntimeData():
    def __init__(self):
        return

"""
    A class used to hold profile data generated in a test run performed by a model worker.
"""
class ProfileData():
    def __init__(self):
        return

"""
    Profile data records in format of time bins within a period of time 'time_limit'.
    Each time bin records' initial content is set to 'init'.
    'bin_size' is set to PLOTTING_BIN_SIZE by default.
"""
class TimeBinRecord():
    def __init__(self, init, time_limit, bin_size = PLOTTING_BIN_SIZE):
        self.bin_size = bin_size
        self.records = [init for _ in timebin_ranges(bin_size, time_limit)]
    
    def index(self, time_stamp):
        return int(time_stamp / self.bin_size)

    def get(self, time_stamp):
        return self.records[self.index(time_stamp)]

    ### Start of commonly used functions for bin records where each record is a list.
    """
        Append 'data' to the corresponding bin record.
    """
    def append_to_bin(self, time_stamp, data):
        self.records[self.index(time_stamp)].append(data)
    
    """
        Collapse collected bin records into count, average & max list.
    """
    def collapse(self):
        count_list = [len(x) for x in self.records]
        avg_list = [np.average(x) if x else 0 for x in self.records]
        max_list = [np.max(x) if x else 0 for x in self.records]
        return count_list, avg_list, max_list
    
    def add(self, time_stamp, data):
        self.records[self.index(time_stamp)] += data

    def append_timestamps(self, data = None):
        if(not data):
            data = self.records
        return bin_records_append_timestamps(data, self.bin_size)

class RecordCounter():
    def __init__(record_bins = False, time_limit = None, bin_size = None):
        self.counter = 0
        self.record_bins = None
        if(record_bins):
            self.record_bins = TimeBinRecord(init = [], time_limit = time_limit, bin_size = bin_size)

    def _operate(self, num, timestamp = None):
        self.counter += num
        if(self.record_bins):
            self.record_bins.append_to_bin(timestamp, self.counter)

    def add(self, timestamp = None):
        self._operate(1, timestamp)
    
    def rm(self, timestamp = None):
        self._operate(-1, timestamp)
    
    def append_timestamps(self):
        return self.record_bins.append_timestamps()

"""
    A class used to hold profiled data specifically collected for a scheduler that is generated in a test run performed by a model worker.
"""
class SchedulerData():
    def __init__(self, scheduler_type):
        self.scheduler_type = scheduler_type
        return

"""
    Monitors the running dcgm processes dcgm_procs, and report the process ids through conn.
    If dcgm processes are not detected (died), start a new one in replace. 
    Upon receiving message from conn, report the final pids of the processes through conn and stop the monitoring process. 
"""
def dcgm_monitor(dcgm_procs, conn):
    while(True):
        if conn.poll():
            msg = conn.recv()
            conn.send([pid for pid, cmd in dcgm_procs])
            break
        time.sleep(5)
        for id, dcgm_proc in enumerate(dcgm_procs):
            pid, cmd = dcgm_proc
            try:
                os.kill(pid, 0)  # Doesn't actually kill; just checks permission/existence
            except ProcessLookupError:
                os.system(cmd + " append")
                with open('tmp.txt', 'r') as pid_file:
                    pid = int(pid_file.read())
                dcgm_procs[id] = (pid, cmd)
            except PermissionError:
                pass
            else:
                pass

"""
    Stop dcgm process through process connection.
"""
def stop_dcgm(conn):
    if(conn is None):
        return
    conn.send("stop")
    final_pids = conn.recv()
    print("Final PID list:", final_pids)
    for pid in final_pids:
        command = "kill -9 {}".format(str(pid))
        os.system(command)

"""
    Given a profile_filepath pointing to a file generated by dcgm, 
    get the data inside and return a list containing the data.
"""
def get_dcgm_data(profile_filepath):
    values = []
    with open(profile_filepath, 'r') as profile_file:
        lines = profile_file.readlines()
        for line in lines:
            line = line.strip('\n').strip(' ')
            if line and not line.startswith("#Entity") and not line.startswith("ID"):
                # print(line, line.split("      "))
                _, value = line.split("      ")
                values.append(float(value))
    return values


"""
    Given a profile_filepath pointing to a file generated by dcgm, 
    get the mean value of the profiled metric.
"""
def get_dcgm_analysis(profile_filepath):
    values = get_dcgm_data(profile_filepath)
    return np.mean(values) if values else None

"""
    Given a profile_filepath pointing to a file generated by dcgm, 
    get the mean value of the profiled metric excluding zero values.
"""
def get_dcgm_analysis_excluding_zeros(profile_filepath):
    values = get_dcgm_data(profile_filepath)
    filtered = [v for v in values if v != 0]
    return np.mean(filtered) if filtered else None

def timebin_ranges(time_bin_len, time_len):
    return np.arange(0, time_len + 5, time_bin_len)


"""
    A max heap with inputs as **(key, value)** pairs (key for sorting and value to be retrieved).
    When value is float allow specification to format the value to several digits by specifiying float_fmt.
    Supports retrieving the sum of all keys in heap.
"""
class MaxHeap:
    def __init__(self, float_fmt=".5f"):
        self.heap = []               # stores (-key, key, str_val, float_val)
        self.float_fmt = float_fmt
        self.total = 0
    
    def __len__(self):
        return len(self.heap)
    
    def get_total(self):
        return self.total

    def _normalize(self, t):
        key = t[0]
        val = t[1]
        str_val = format(val, self.float_fmt)
        return key, str_val

    def push(self, t):
        key, str_val = self._normalize(t)
        self.total += key
        heapq.heappush(self.heap, (-key, str_val))

    def top(self):
        key, str_val = self.heap[0]
        key = -minus_key
        return (key, float(str_val))

    def pop(self):
        minus_key, str_val = heapq.heappop(self.heap)
        key = -minus_key
        self.total -= key
        return (key, float(str_val))


"""
    A max heap which is deletable, and the inputs are **(key, value)** pairs (key for sorting and value to be retrieved).
    When value is float allow specification to format the value to several digits by specifiying float_fmt.
"""
class DeletableMaxHeap:
    def __init__(self, float_fmt=".5f"):
        self.heap = []               # stores (-key, key, str_val, float_val)
        self.counter = Counter()
        self.float_fmt = float_fmt
    
    def __len__(self):
        return len(self.heap)

    def _normalize(self, t):
        key = t[0]
        val = t[1]
        str_val = format(val, self.float_fmt)
        return key, str_val

    def push(self, t):
        key, str_val = self._normalize(t)
        heapq.heappush(self.heap, (-key, str_val))
        self.counter[(-key, str_val)] += 1

    def delete(self, t):
        key, str_val = self._normalize(t)
        self.counter[(-key, str_val)] -= 1

    def _clean_top(self):
        while self.heap and self.counter[(self.heap[0][0], self.heap[0][1])] == 0:
            heapq.heappop(self.heap)

    def top(self):
        self._clean_top()
        if not self.heap:
            return None
        key, str_val = self.heap[0]
        return (-key, float(str_val))

    def pop(self):
        self._clean_top()
        if not self.heap:
            return None
        key, str_val = heapq.heappop(self.heap)
        self.counter[(key, str_val)] -= 1
        return (-key, float(str_val))
