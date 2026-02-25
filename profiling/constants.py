import os
import csv
import copy
import time
import heapq
import subprocess
import numpy as np
from enum import Enum
from typing import List
from collections import Counter
from plotting.constants import PLOTTING_BIN_SIZE

EPS = 1e-6
DCGM_TMP_FILE = "dcgm_pid.log"

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

def get_cmd_output(cmd):
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, text=True)
    stdout, stderr = p.communicate()
    return stdout

def device_init(all_devices : List[str], device_ids : List[str], mps = None):
    using_mps = get_cmd_output(f"gmanager checkmps {',' .join(all_devices)}").lower().strip() == "true"
    if(using_mps):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(all_devices.index(device_id)) for device_id in device_ids])
        UUID_hash = get_cmd_output(f"gmanager getuuid_hash {','.join(all_devices)}").strip()
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-{UUID_hash}" 
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-mps-{UUID_hash}"
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps) if (mps is not None) else '100'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)

def background(cmd, log):
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
    Special cases:
        - If string list is empty, first empty list is generated.
        - In string, if empty term is given ('x1,'); the empty term is set to None.
    If argstype is int, the list will be a int list;
    If given expend (int), it requires the list to be expanded into [x1, x2, ..., x_{expand}]
    If default is given, expand list with default; else expand with x1 (or None if x1 non-existent). 
"""
def get_comma_list(strlist, argstype, expand = None, **kwargs):
    results = strlist.strip("\n").split(",") if strlist else []
    if(argstype == ArgsType.INT):
        results = [int(result) if result != '' else None for result in results]
    else:
        results = [result if result != '' else None for result in results]
    if(expand is not None):
        if(len(results) != expand):
            default = kwargs.get("default", results[0] if len(results) else None)
            for i in range(len(results), expand):
                results.append(default)
    return results

"""
    Same functionality as get_comma_list↑, only with 'x1' return x1 instead of [x1]
"""
def get_comma_list_or_single(strlist, argstype, expand = None, **kwargs):
    comma_list = get_comma_list(strlist, argstype, expand, **kwargs)
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
    It is also responsible for filling in missing data with defaults which are required for successful execution:
        currently - batchsize, seq_len
    batch size: /
    no sync: True / False, if set to true always get a input tensor of batch size 1 
            (input data will be manually submitted up to batch size times at a time without synchronizing the GPU)
    inter_flag: True / False, if set to true means get a partial input
    start: id, indicating which partial result to fetch. get the result generated after executing the (id - 1) operators
         (operator count start from 0)
"""
class InputConfig():
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", None)
        if(not self.batch_size):
            self.batch_size = 1
        self.nosync = kwargs.get("nosync", None)
        self.inter_flag = kwargs.get("inter_flag", None)
        self.start = kwargs.get("start", None)
        self.seq_len = kwargs.get("seq_len", None)
        if(not self.seq_len):
            self.seq_len = 1

"""
    A class used to pack all configs related to a running work for the worker to perform.
"""
class TestConfig():
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 1)
        self.start = kwargs.get("start", None)
        self.end = kwargs.get("end", None)
        self.global_tag = kwargs.get("global_tag", None)
        self.dcgm_group = kwargs.get("dcgm_group", -1)
        self.seq_len = kwargs.get("seq_len", None)
        self.nosync = kwargs.get("nosync", False)
        self.trace_path = kwargs.get("trace_path", None)
        self.set_recording_action(kwargs.get("recording_action", RecordingAction.NO_RECORD))
        self.recording_action_saved = None
        self.scheduler = kwargs.get("scheduler", None)
        self.scheduler_saved = None
        self.global_cnt = kwargs.get("global_cnt", 1)
        self.inter_flag = kwargs.get("inter_flag", False)

        self.test_time = kwargs.get("test_time", None)
        self.test_time_default = self.test_time is None
        if(self.test_time is None):
            self.test_time = 100
    
    def set_recording_action(self, recording_action):
        self.recording_action = recording_action
        self.record_statistics = (self.recording_action in RECORDING_ACTION_MAP["statistics"])
        self.record_dcgm = (self.recording_action in RECORDING_ACTION_MAP["dcgm"])
    
    def set_warmup(self):
        self.recording_action_saved = self.recording_action
        self.scheduler_saved = self.scheduler
        self.scheduler = None
        self.set_recording_action(RecordingAction.NO_RECORD)
    
    def set_test(self):
        if(self.recording_action_saved):
            self.set_recording_action(self.recording_action_saved)
            self.recording_action_saved = None
        if(self.scheduler_saved):
            self.scheduler = self.scheduler_saved
            self.scheduler_saved = None

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
    A class used to hold profiled data specifically collected for a scheduler that is generated in a test run performed by a model worker.
"""
class SchedulerData():
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
        self.time_limit = time_limit
        self.records = [copy.copy(init) for _ in timebin_ranges(bin_size, time_limit)]
    
    """
        Get TimeBinRecord object through a given list of format: [(time stamp, data), ...].
        This created TimeBinRecord object is a summed bin. 
    """
    @classmethod
    def get_summed_time_bin_record(cls, data_records, bin_size = PLOTTING_BIN_SIZE):
        time_bin_record = TimeBinRecord(init = 0, time_limit = max([x[0] for x in data_records]), bin_size = bin_size)
        for record in data_records:
            time_bin_record.add(*record)
        return time_bin_record

    def index(self, time_stamp):
        idx = int(time_stamp / self.bin_size)
        if(idx >= len(self.records)):
            return None
        return idx

    def get(self, time_stamp):
        idx = self.index(time_stamp)
        if(idx is None):
            print("Warning: accessing time bin data out of time limit.") 
            return 0
        return self.records[idx]
    
    """
        Collapse collected bin records into count, average & max list.
        Applicable if bin records are lists.
    """
    def collapse(self):
        count_list = [len(x) for x in self.records]
        avg_list = [np.average(x) if x else 0 for x in self.records]
        max_list = [np.max(x) if x else 0 for x in self.records]
        return count_list, avg_list, max_list
    
    """
        Append 'data' to the corresponding bin record.
    """
    def append_to_bin(self, time_stamp, data):
        idx = self.index(time_stamp)
        if(idx is not None):
            self.records[idx].append(data)
    
    def add(self, time_stamp, data):
        idx = self.index(time_stamp)
        if(idx is not None):
            self.records[idx] += data

    def append_timestamps(self, data = None):
        if(not data):
            data = self.records
        return bin_records_append_timestamps(data, self.bin_size)
    
    def update(self, other, operation):
        if ((not isinstance(other, TimeBinRecord)) or self.time_limit != other.time_limit or self.bin_size != other.bin_size):
            raise Exception("Cannot add these two objects (the other is not TimeBinRecord / shape does not match)")
        match operation:
            case "sum":
                for idx in range(len(self.records)):
                    self.records[idx] += other.records[idx]
            case "extend":
                for idx in range(len(self.records)):
                    self.records[idx].extend(other.records[idx])
            case _:
                raise Exception(f"Update operation {operation} not supported for TimeBinRecords")

"""
    A record counter that can be associated with a record_bin (TimeBinRecord).
    The counter tracks the current count, and the record_bin records how the count changes over time.
"""
class RecordCounter():
    def __init__(self, record_bins = None):
        self.counter = 0
        self.record_bins = record_bins

    def _operate(self, num, timestamp = None):
        self.counter += num
        if(self.record_bins):
            self.record_bins.append_to_bin(timestamp, self.counter)

    def add(self, timestamp = None):
        self._operate(1, timestamp)
    
    def rm(self, timestamp = None):
        self._operate(-1, timestamp)
    
    def set_record_bins(self, record_bins):
        self.record_bins = record_bins

"""
    Monitors the running dcgm processes dcgm_procs, and report the process ids through conn.
    (The dcgm processes monitors devices included in the dcgm group with group id $dcgm_group_id)
    If dcgm processes are not detected (died), start a new one in replace. 
    Upon receiving message from conn, report the $dcgm_group_id and final pids of the processes through conn and stop the monitoring process. 
"""
def dcgm_dmon_monitor(dcgm_group_id, dcgm_procs, conn):
    while(True):
        if conn.poll():
            msg = conn.recv()
            conn.send(dcgm_group_id)
            conn.send([pid for pid, cmd in dcgm_procs])
            break
        time.sleep(5)
        for id, dcgm_proc in enumerate(dcgm_procs):
            pid, cmd = dcgm_proc
            try:
                os.kill(pid, 0)  # Doesn't actually kill; just checks permission/existence
            except ProcessLookupError:
                os.system(cmd + " append")
                with open(DCGM_TMP_FILE, 'r') as pid_file:
                    pid = int(pid_file.read())
                dcgm_procs[id] = (pid, cmd)
            except PermissionError:
                pass
            else:
                pass

"""
    Stop dcgm process through process connection.
    The dcgm group is created for temporal use, so delete operation for clean up is also performed.
"""
def stop_dcgm(conn):
    if(conn is None):
        return
    conn.send("stop")
    dcgm_group_id = conn.recv()
    final_pids = conn.recv()
    print("Final PID list:", final_pids)
    for pid in final_pids:
        command = "kill -9 {}".format(str(pid))
        os.system(command)
    os.system(f'gmanager dcgm_delg {dcgm_group_id}')

"""
    Given a profile_filepath pointing to a file generated by dcgm, 
    get the data inside and return a list containing the data.
"""
def get_dcgm_data(profile_filepath):
    values = []

    device_records = {}
    value_tmp = 0
    value_cnt = 0
    
    with open(profile_filepath, 'r') as profile_file:
        lines = profile_file.readlines()
        for line in lines:
            line = line.strip('\n').strip(' ')
            if line and not line.startswith("#Entity") and not line.startswith("ID"):
                device, value = line.split("      ")
                if(device in device_records and value_cnt):
                    values.append(value_tmp / value_cnt)
                    device_records = {}
                    value_tmp = 0
                    value_cnt = 0
                value_tmp += float(value)
                value_cnt += 1
                device_records[device] = True
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
    return np.arange(0, time_len + 0.5, time_bin_len)

def write_list_to_csv(file_path, headers, record_list):
    with open(file_path, "w", newline="", encoding="utf-8") as log_output:
        writer = csv.writer(log_output)
        writer.writerow(headers)
        for record in record_list:
            if(isinstance(record, list) or isinstance(record, tuple)):
                writer.writerow(record)  # Writes entire list
            else:
                writer.writerow([record])


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
        minus_key, str_val = self.heap[0]
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
