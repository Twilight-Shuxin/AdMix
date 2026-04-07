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
from dataclasses import dataclass, field
from multiprocessing import Process, Pipe
from plotting.constants import PLOTTING_BIN_SIZE

EPS = 1e-6

class ArgsType(Enum):
    INT = 1
    STRING = 2

@dataclass
class RecordingAction():
    record_statistics: bool = False
    record_dcgm: bool = False
    record_pw: bool = False

"""
    Runtime Mode of profiler.
"""
class RuntimeMode(Enum):
    TEST = "TEST" # Original mode with all free functionalities
    PROFILE = "PROFILE" # Profile single model worker
    SERVE = "SERVE" # Serve LLM with other models utilizing scheudler (None:no specific scheduler)

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
        self.recording_action = kwargs.get("recording_action", RecordingAction())
        self.recording_action_saved = None
        self.scheduler = kwargs.get("scheduler", None)
        self.scheduler_saved = None
        self.global_cnt = kwargs.get("global_cnt", 1)
        self.inter_flag = kwargs.get("inter_flag", False)

        self.test_time = kwargs.get("test_time", None)
        self.test_time_default = self.test_time is None
        if(self.test_time is None):
            self.test_time = 100
    
    def set_warmup(self):
        self.recording_action_saved = self.recording_action
        self.scheduler_saved = self.scheduler
        self.scheduler = None
        self.recording_action = RecordingAction()
    
    def set_test(self):
        if(self.recording_action_saved):
            self.recording_action = self.recording_action_saved
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

def get_dcgm_file_path(log_path, tag, profile_statistic):
    return os.path.join(log_path, "-".join([tag, str(profile_statistic)]) + ".log")

def create_dcgm_group(devices):
    p = subprocess.Popen(["gmanager dcgm_addg tmp"], shell=True, stdout=subprocess.PIPE, text=True)
    stdout, _ = p.communicate()
    group_id = int(stdout.strip())
    os.system(f"gmanager dcgm_add2g {group_id} {' '.join(list(map(str, devices)))}")
    return group_id

def _launch_dcgm_dmon(group_id, profile_id, log_filepath=None, append=False):
    """
    Launch a dcgmi dmon process for the given group and profile field ID.
    If log_filepath is given, stdout is written to that file (appended if append=True);
    otherwise stdout is captured via PIPE for in-memory collection.
    Returns the Popen object.
    """
    if log_filepath:
        stdout_target = open(log_filepath, 'a' if append else 'w')
    else:
        stdout_target = subprocess.PIPE
    proc = subprocess.Popen(["dcgmi", "dmon", "-g", str(group_id), "-e", str(profile_id)],
                            stdout=stdout_target, stderr=subprocess.DEVNULL, text=True)
    if log_filepath:
        stdout_target.close()
    return proc

def _get_dcgm_launch_args(group_id, profile_statistic, log_path=None, tag=None):
    """
    Build the launch args tuple (group_id, profile_id, log_filepath) for a single
    profile statistic. log_filepath is None if log_path is not provided.
    """
    profile_id = profile_dict[profile_statistic]
    log_filepath = get_dcgm_file_path(log_path, tag, profile_statistic) if log_path else None
    return (group_id, profile_id, log_filepath)

def _launch_dcgm_watchdog(group_id, launch_args_list):
    """
    Spawn the dcgm_dmon_monitor watchdog Process and return the parent connection.
    launch_args_list is a list of (group_id, profile_id, log_filepath) tuples — one per
    statistic to monitor. Popen objects are created inside the child process to avoid
    pickling issues with the spawn start method.
    """
    parent_conn, child_conn = Pipe()
    Process(target=dcgm_dmon_monitor, args=(group_id, launch_args_list, child_conn)).start()
    return parent_conn

def start_dcgm(dcgm_metric_group, devices, log_path=None, tag=None, dcgm_group_id=None):
    """
    Start dcgm monitoring for all profile statistics in dcgm_metric_group.
    Creates a temporary dcgm group (unless dcgm_group_id is provided) and starts one
    dcgmi dmon process per statistic in profile_group_dict[dcgm_metric_group].
    Returns a connection to the watchdog process, or None if dcgm_metric_group == -1.
    """
    if(dcgm_metric_group == -1):
        return None
    if(dcgm_group_id is None):
        dcgm_group_id = create_dcgm_group(devices)
    launch_args_list = [_get_dcgm_launch_args(dcgm_group_id, ps, log_path, tag) for ps in profile_group_dict[dcgm_metric_group]]
    return _launch_dcgm_watchdog(dcgm_group_id, launch_args_list)

def start_dcgm_single(profile_statistic, devices, log_path=None, tag=None, dcgm_group_id=None):
    """
    Start dcgm monitoring for a single profile statistic.
    Creates a temporary dcgm group (unless dcgm_group_id is provided) and starts one
    dcgmi dmon process for the given profile_statistic.
    Returns a connection to the watchdog process.
    """
    if(dcgm_group_id is None):
        dcgm_group_id = create_dcgm_group(devices)
    return _launch_dcgm_watchdog(dcgm_group_id, [_get_dcgm_launch_args(dcgm_group_id, profile_statistic, log_path, tag)])

"""
    Monitors the running dcgm processes dcgm_procs, and report the process ids through conn.
    (The dcgm processes monitors devices included in the dcgm group with group id $dcgm_group_id)
    If dcgm processes are not detected (died), start a new one in replace.
    Upon receiving message from conn, report the $dcgm_group_id and final pids of the processes through conn and stop the monitoring process.
"""
def dcgm_dmon_monitor(dcgm_group_id, launch_args_list, conn):
    dcgm_procs = [(launch_args, _launch_dcgm_dmon(*launch_args)) for launch_args in launch_args_list]
    accumulated = [[] for _ in dcgm_procs]
    while(True):
        if conn.poll(5): # blocks for 5 seconds but respond immediately if received message
            conn.recv()
            conn.send(dcgm_group_id)
            conn.send([proc.pid for _, proc in dcgm_procs])
            outputs = []
            for i, (launch_args, proc) in enumerate(dcgm_procs):
                _, _, log_filepath = launch_args
                proc.terminate()
                if log_filepath:
                    proc.wait()
                    outputs.append(None)
                else:
                    stdout, _ = proc.communicate()
                    accumulated[i].append(stdout)
                    outputs.append("".join(accumulated[i]))
            conn.send(outputs)
            break
        for i, (launch_args, proc) in enumerate(dcgm_procs):
            if proc.poll() is not None:
                _, _, log_filepath = launch_args
                if not log_filepath:
                    stdout, _ = proc.communicate()
                    accumulated[i].append(stdout)
                new_proc = _launch_dcgm_dmon(*launch_args, append=True)
                dcgm_procs[i] = (launch_args, new_proc)

"""
    Background watchdog process for power monitoring (analogous to dcgm_dmon_monitor).
    Starts nvidia-smi, monitors it, and restarts it if it dies.
    Upon receiving "stop" via conn, terminates nvidia-smi, sends back collected output (or None
    if redirected to a file), and exits.
    On restart with redirect_output, the file is opened in append mode to preserve prior data.
"""
def power_monitor_watchdog(devices, redirect_output, conn):
    cmd = ["nvidia-smi", "--query-gpu=index,power.draw", "--format=csv,noheader,nounits", "-l", "1", f"--id={','.join(devices)}"]
    stdout_target = open(redirect_output, 'w') if redirect_output else subprocess.PIPE
    monitor_proc = subprocess.Popen(cmd, stdout=stdout_target, stderr=subprocess.PIPE, text=True)
    accumulated = []
    while True:
        if conn.poll(5): # blocks for 5 seconds but respond immediately if received message
            conn.recv()
            monitor_proc.terminate()
            if redirect_output:
                monitor_proc.wait()
                stdout_target.close()
                conn.send(None)
            else:
                stdout, _ = monitor_proc.communicate()
                accumulated.append(stdout)
                conn.send("".join(accumulated))
            break
        if monitor_proc.poll() is not None:
            if redirect_output:
                stdout_target.close()
                stdout_target = open(redirect_output, 'a')
            else:
                stdout, _ = monitor_proc.communicate()
                accumulated.append(stdout)
            monitor_proc = subprocess.Popen(cmd, stdout=stdout_target, stderr=subprocess.PIPE, text=True)

"""
    Start power monitoring with a watchdog process (analogous to start_dcgm).
    Returns a connection to the watchdog process.
"""
def start_power_monitor(devices, redirect_output=None):
    parent_conn, child_conn = Pipe()
    watchdog = Process(target=power_monitor_watchdog, args=(devices, redirect_output, child_conn))
    watchdog.start()
    return parent_conn

"""
    Stop power monitoring (analogous to stop_dcgm).
    Returns collected stdout text if no redirect_output was given, else None.
"""
def stop_power_monitor(conn):
    if conn is None:
        return None
    conn.send("stop")
    return conn.recv()

def delete_dcgm_group(dcgm_group_id):
    os.system(f'gmanager dcgm_delg {dcgm_group_id}')

"""
    Stop dcgm process through process connection.
    The dcgm group is created for temporal use, so delete operation for clean up is also performed.
"""
def stop_dcgm(conn, del_dcgm_group_flag = True):
    if(conn is None):
        return None
    conn.send("stop")
    dcgm_group_id = conn.recv()
    final_pids = conn.recv()
    outputs = conn.recv()
    if(del_dcgm_group_flag):
        delete_dcgm_group(dcgm_group_id)
    return outputs

"""
    A list of records, with record format as: (device0, data).
    Normally, the monitor returns per fixed time interval with one data item for each device.
    This function averages the monitered data across all devices and combine them into one time series. 
"""
def get_device_monitored_data_avg(records):
    device_records = {}
    value_tmp = 0
    value_cnt = 0
    avg_values = []
    for device, value in records:
        if(device in device_records and value_cnt):
            avg_values.append(value_tmp / value_cnt)
            device_records = {}
            value_tmp = 0
            value_cnt = 0
        value_tmp += float(value)
        value_cnt += 1
        device_records[device] = True
    if value_cnt:
        avg_values.append(value_tmp / value_cnt)
    return avg_values

"""
    Parse output generated by dcgm and return records: (device, value)
"""
def parse_dcgm_output(text: str):
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#Entity") and not line.startswith("ID"):
            try:
                device, value = line.split("      ")
                records.append((device, float(value)))
            except ValueError:
                pass
    return records

"""
    Given a profile_filepath pointing to a file generated by dcgm,
    get the data inside and return a tuple:
        1. overall average; 2. a list containing the time series data (averaged across all devices).
"""
def get_dcgm_data(profile_filepath):
    with open(profile_filepath, 'r') as profile_file:
        records = parse_dcgm_output(profile_file.read())
    values = get_device_monitored_data_avg(records)
    return (np.mean(values) if values else None, values)

def parse_power_output(text):
    records = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            try:
                device, value = line.split(",")
                records.append((device, float(value.strip())))
            except ValueError:
                pass
    return records

"""
    Given a profile_filepath pointing to a file generated by nvidia-smi power.draw,
    get the data inside and return a tuple:
        1. overall average; 2. a list containing the time series data (averaged across all devices).
"""
def get_power_data(profile_filepath):
    with open(profile_filepath, 'r') as profile_file:
        records = parse_power_output(profile_file.read())
    values = get_device_monitored_data_avg(records)
    return (np.mean(values) if values else None, values)

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
