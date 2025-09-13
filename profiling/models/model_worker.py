from torch.multiprocessing import Process
from profiling.constants import *
import torch
import time
import numpy as np
from trace.trace_utils import Trace
import os
from profiling.scheduling.monitor import *
from profiling.scheduling.utils import schedulers
from plotting.constants import *
import csv
import json

def read_intensity_from_pipe():
    with open(PIPE_PATH, "r") as fifo:
        latest_value = fifo.readlines()[-1]  # Get the last updated value
    return latest_value  

def get_input_config_from_test_config(test_config):
    return InputConfig(test_config.batch_size, nosync=test_config.nosync, start=test_config.start, 
        inter_flag=test_config.inter_flag) 

# device is limited to CUDA_VISIBLE_DEVICE set before worker creation, thus all worker only see 'cuda'
class ModelWorker(Process):
    def __init__(self, model_name, device_id, mps, pipe, barrier, log_path, profile_path):
        super().__init__()
        self.model_name = model_name
        self.mps = mps
        self.pipe = pipe
        self.single_barrier, self.barrier = barrier
        self.log_path = log_path
        self.profile_path = profile_path
        self.logs = []
        self.device = "cuda"
        self.true_device_id = device_id
        self.supported_batchsize = [x for x in range(1,33)]
        self.dirname, filename = os.path.split(os.path.abspath(__file__))
        self.warmup_time = 10
        self.test_time = 300

    def serve():
       raise NotImplementedError
    
    def test():
        raise NotImplementedError
    
    def shutdown(self):
        return
    
    def get_trace(self, trace_path):
        self.trace = Trace(trace_path)
    
    def get_dcgm_file_path(self, test_config, profile_statistic):
        return os.path.join(self.log_path, "-".join([test_config.tag, str(profile_statistic)]) + ".log")
    
    def statistics_recording_init(self, test_config):
        self.logs = []
        time_len = self.test_time if test_config.test_time == None else test_config.test_time
        self.absolute_throughput_time_bins = [0 for x in np.arange(0, time_len + PLOTTING_BIN_SIZE * 5, PLOTTING_BIN_SIZE)]
        self.absolute_throughput = 0
        self.throughput = 0

    def start_dcgm(self, test_config):
        if(test_config.dcgm_group == None):
            return []
        pids = []
        for profile_statistic in profile_group_dict[test_config.dcgm_group]:
            profile_id = profile_dict[profile_statistic]
            log_filepath = self.get_dcgm_file_path(test_config,profile_statistic)
            command = "{}/dcgm_monitor.sh {} {} \"{}\"".format(self.dirname, str(self.true_device_id + 2), str(profile_id), 
                log_filepath)
            os.system(command)
            with open('tmp.txt', 'r') as pid_file:
                pid = int(pid_file.read())
            pids.append(pid)
        return pids
    
    def get_tag(self, test_config):
        tag = self.model_name
        sub_tag = ""
        if(test_config.batch_size != None):
            sub_tag += str(test_config.batch_size)
        if(self.mps != None and self.mps != 100):
            if (len(sub_tag) != 0):
                sub_tag += "_"
            sub_tag += "m" + str(self.mps)
        if(len(sub_tag) != 0):
            sub_tag = "(" + sub_tag + ")"
        test_config.get_inter_tag(self)
        if(test_config.inter_flag):
            sub_tag += "[" + str(test_config.start) + "," + str(test_config.end) + "]"
        return tag + sub_tag
            
    def get_input(self, input_config):
        batch_size = input_config.batch_size
        if(input_config.nosync):
            batch_size = 1
        if input_config.inter_flag:
            return self.partial_input[batch_size][input_config.start]
        else:
            return self.random_input[batch_size]
    
    def _inference(self, input_data, start, end):
        if(start != 0 or end != self.total_module): 
            result = input_data
            for i in range(start, end, PARTIAL_MODEL_STRIDE):
                result = self.partial_model[i](result)
        else:
            result = self.model(input_data)
        return result

    def inference(self, input, start, end, sync = False, rec = False):
        start_time = time.perf_counter()
        self._inference(input, start, end)
        if(sync): 
            torch.cuda.synchronize()
        self.logs.append([input.shape[0], (time.perf_counter() - start_time) * 1000])
            
    def get_file_path(self, test_config, file_name, subpath_name = None):
        if(subpath_name != None):
            print(self.log_path, subpath_name,  "-".join([test_config.tag, file_name]))
            print("-->", os.path.join(self.log_path, subpath_name, "-".join([test_config.tag, file_name])))
            return os.path.join(self.log_path, subpath_name, "-".join([test_config.tag, file_name]))
        return os.path.join(self.log_path, "-".join([test_config.tag, file_name]))
    
    def write_analytics(self, test_config, file_name, record_list):
        log_filepath = self.get_file_path(test_config, file_name)
        if(len(record_list) == 0):
            return
        with open(log_filepath, "w+") as log_output:
            log_output.write("mean " + str(np.average(record_list)) + "\n")
            log_output.write("min " + str(np.amin(record_list)) + "\n")
            log_output.write("max " + str(np.amax(record_list)) + "\n")
            log_output.write("var " + str(np.var(record_list)) + "\n")
            log_output.write("90% " + str(np.percentile(record_list, 90)) + "\n")
            log_output.write("95% " + str(np.percentile(record_list, 95)) + "\n")
            log_output.write("99% " + str(np.percentile(record_list, 99)) + "\n")
    
    def write_all(self, test_config, file_name, record_list):
        log_filepath = self.get_file_path(test_config, file_name)
        with open(log_filepath, "w+") as log_output:
            for log_record in record_list: 
                log_output.write(list_to_string(log_record) + "\n")
    
    def write_list_to_csv(self, test_config, file_name, headers, record_list, subpath_name = None):
        log_filepath = self.get_file_path(test_config, file_name, subpath_name)
        with open(log_filepath, "w", newline="", encoding="utf-8") as log_output:
            writer = csv.writer(log_output)
            writer.writerow(headers)
            for record in record_list:
                if(isinstance(record, list)):
                    writer.writerow(record)  # Writes entire list
                else:
                    writer.writerow([record])
    
    def write_log_normal(self, test_config):
        self.write_analytics(test_config, "summary.log", self.logs)
        self.write_list_to_csv(test_config, "all.csv", ['Batch Size', 'Latency (ms)'], self.logs)
        self.write_all(test_config, "flow.log", bin_records_append_timestamps(self.absolute_throughput_time_bins, PLOTTING_BIN_SIZE))
    
    def write_log_trace(self, test_config):
        self.write_analytics(test_config, "summary.log", self.logs[0])
        self.write_list_to_csv(test_config, "all.csv", ['Latency (ms)'], list(map(list, zip(*self.logs))))
        # 0: produced time; 1: processed time
        self.write_all(test_config, "flow.log", bin_records_append_timestamps(self.absolute_throughput_time_bins, PLOTTING_BIN_SIZE))
      
    def check_performace(self, test_config):
        return self.absolute_throughput, self.throughput

    def random_test_period(self, test_config, time_len, rec = False):
        self.start_time_pointer = time.perf_counter()
        input = self.get_input(get_input_config_from_test_config(test_config))
        while(time.perf_counter() - self.start_time_pointer < time_len):
            if(test_config.nosync):
                start_time_sub = time.perf_counter()
                for i in range(test_config.batch_size):
                    self.inference(input, test_config.start, test_config.end, False, False)
                torch.cuda.synchronize()
                if(rec):
                    self.logs.append((time.perf_counter() - start_time_sub) * 1000)
                    self.absolute_throughput += test_config.batch_size
                    self.absolute_throughput_time_bins[int((time.perf_counter() - self.start_time_pointer) / PLOTTING_BIN_SIZE)] += test_config.batch_size
                    self.throughput += 1
            else:
                self.inference(input, test_config.start, test_config.end, True, rec in RECORDING_ACTION_MAP["statistics"])
                if(rec):
                    self.absolute_throughput += test_config.batch_size
                    self.absolute_throughput_time_bins[int((time.perf_counter() - self.start_time_pointer) / PLOTTING_BIN_SIZE)] += test_config.batch_size
                    self.throughput += 1

    def trace_test_period(self, test_config, time_len, rec = False):
        self.start_time_pointer = time.perf_counter()
        self.logs = [[], []]
        time_limit = self.start_time_pointer + time_len
        input = self.get_input(get_input_config_from_test_config(test_config))
        while(True):
            req_time = self.trace.next()
            sleep_time = req_time + self.start_time_pointer - time.perf_counter()
            if(sleep_time > 0):
                time.sleep(sleep_time)
            if(time.perf_counter() > time_limit):
                break
            req_start_time = time.perf_counter()
            self.inference(input, test_config.start, test_config.end, True, False)
            torch.cuda.synchronize()
            req_process_time = time.perf_counter() - req_start_time
            req_produced_time = time.perf_counter() - req_time - self.start_time_pointer
            if(rec):
                self.logs[0].append(req_process_time * 1000)
                self.logs[1].append(req_produced_time * 1000)
                self.absolute_throughput += test_config.batch_size
                self.absolute_throughput_time_bins[int((time.perf_counter() - self.start_time_pointer) / PLOTTING_BIN_SIZE)] += self.batch_size
                self.throughput += 1
    
    def random_test(self, test_config):
        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        recording_action = RecordingAction.ALL if test_config.recording_action == None else test_config.recording_action
        record_statistics = (recording_action in RECORDING_ACTION_MAP["statistics"])
        if(record_statistics):
            self.statistics_recording_init(test_config)

        if(recording_action in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        
        if(test_config.trace_path != None):
            self.trace_test_period(test_config, test_time, record_statistics)
        else:
            self.random_test_period(test_config, test_time, record_statistics)

        if(recording_action in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)
        if(record_statistics):
            if(test_config.trace_path != None):
                self.write_log_trace(test_config)
            else:
                self.write_log_normal(test_config)
    
    def collect_dcgm(self, test_config):
        dcgm_data = {}
        for dcgm_group in test_config.dcgm_group:
            for profile_statistic in profile_group_dict[dcgm_group]:
                dcgm_data[profile_statistic] = get_dcgm_analysis(self.get_dcgm_file_path(test_config, profile_statistic))
        return dcgm_data

    def common_start(self):
        self.submodules = self.model.get_submodules()
        self.total_module = len(self.submodules)
        # self.partial_model = {}
        # self.partial_input = {}
        # for batch_size in self.supported_batchsize:
        #     self.partial_input[batch_size] = {}
        #     self.partial_input[batch_size][0] = self.random_input[batch_size]
        # for i in range(0, self.total_module, PARTIAL_MODEL_STRIDE):
        #     right = min(i+PARTIAL_MODEL_STRIDE, self.total_module)
        #     self.partial_model[i] = torch.nn.Sequential(*self.submodules[i:right]).cuda(self.device)
        #     if(right == self.total_module):
        #         break
        #     for batch_size in self.supported_batchsize:
        #         partial_input_current = self.partial_input[batch_size][i]
        #         self.partial_input[batch_size][right] = self._inference(partial_input_current, i, right)
    
    def warmup(self):
        warmup_config = TestConfig(1, None, self.supported_batchsize[-1], 0, self.total_module, None, None)
        warmup_config.fit(self)
        self.random_test_period(warmup_config, 5)
    
    def profile(self):
        max_iter = 20
        profile_data = {}
        profile_data["model_name"] = self.model_name
        for batch_size in self.supported_batchsize:
            spent_time = []
            for id, iter in enumerate(range(max_iter)):
                start_time = time.perf_counter()
                self.inference(self.random_input[batch_size], 0, self.total_module, True, False)
                if(id > 5):
                    spent_time.append(time.perf_counter() - start_time)
            profile_data[batch_size] = np.percentile(spent_time, 99)
        with open(os.path.join(self.profile_path, self.model_name + ".json"), "w") as profile_file:
            json.dump(profile_data, profile_file, indent=4)
    
    def read_profile_data(self):
        with open(os.path.join(self.profile_path, self.model_name + ".json"), "r") as profile_file:
            self.profile_data = json.load(profile_file)
        return self.profile_data

    def opportunistic_serve(self, test_config):
        global schedulers
        self.scheduler = schedulers[test_config.scheduler]()
        self.scheduling_data_format = schedulers[test_config.scheduler].data_format

        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        rec = RecordingAction.NO_RECORD if test_config.recording_action == None else test_config.recording_action
        if(rec in RECORDING_ACTION_MAP["statistics"]):
            self.statistics_recording_init(test_config)
            
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        self.start_time_pointer = time.perf_counter()
        self.scheduler.serve(self, test_time, rec)
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids) 
        if(rec in RECORDING_ACTION_MAP["statistics"]):
            self.write_log_normal(test_config)

    def run(self) -> None:
        self.init_start()
        self.common_start()
        self.warmup()
        self.barrier.wait()
        while(True):
            action = self.pipe.recv()
            if(action == "profile"):
                self.profile()
                self.single_barrier.wait()
            elif(action == "test"):
                test_config = self.pipe.recv()
                test_config.fit(self)
                self.random_test(test_config)
                self.barrier.wait()
            elif(action == "tag"):
                test_config = self.pipe.recv()
                self.pipe.send(self.get_tag(test_config))
            elif(action == "collect"):
                test_config = self.pipe.recv()
                test_config.get_tag(self)
                if(test_config.dcgm_group != None):
                    self.pipe.send(self.collect_dcgm(test_config))
                self.pipe.send(self.check_performace(test_config))
            elif(action == "opp_serve"):
                test_config = self.pipe.recv()
                test_config.fit(self)
                self.opportunistic_serve(test_config)
                self.barrier.wait()
            elif(action == "stop"):
                break
            else:
                raise NotImplementedError
        self.barrier.wait()
        self.shutdown()
    