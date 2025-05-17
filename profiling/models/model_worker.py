from torch.multiprocessing import Process
from profiling.constants import *
import torch
import time
import numpy as np
from trace.trace_utils import Trace
import os

def read_intensity_from_pipe():
    with open(PIPE_PATH, "r") as fifo:
        latest_value = fifo.readlines()[-1]  # Get the last updated value
    return latest_value  

def get_input_config_from_test_config(test_config):
    return InputConfig(test_config.batch_size, nosync=test_config.nosync, start=
        test_config.start if test_config.inter_flag else None) 

class InputConfig():
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.nosync = kwargs["nosync"] if "nosync" in kwargs else None
        self.inter_flag = kwargs["inter_flag"] if "inter_flag" in kwargs else None
        self.start = kwargs["start"] if "start" in kwargs else None

# device is limited to CUDA_VISIBLE_DEVICE set before worker creation, thus all worker only see 'cuda'
class ModelWorker(Process):
    def __init__(self, model_name, device_id, mps, pipe, barrier, log_path):
        super().__init__()
        self.model_name = model_name
        self.mps = mps
        self.pipe = pipe
        self.barrier = barrier
        self.log_path = log_path
        self.logs = []
        self.device = "cuda"
        self.true_device_id = device_id
        self.supported_batchsize = [1, 2, 4, 8, 16, 32]
        self.dirname, filename = os.path.split(os.path.abspath(__file__))
        self.slo = 100 # currently all are set to 100ms
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

    def start_dcgm(self, test_config):
        if(test_config.dcgm_group == -1):
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
        tag += "(" + str(test_config.batch_size)
        if(self.mps != -1 and self.mps != 100):
            tag += "_m" + str(self.mps)
        tag += ")"
        return tag
            
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
        start_time = time.time()
        self._inference(input, start, end)
        if(sync): 
            torch.cuda.synchronize()
            if(rec):
                self.logs.append((time.time() - start_time) * 1000)
        
    def get_file_path(self, test_config, file_name):
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
    
    def write_log_normal(self, test_config):
        self.write_analytics(test_config, "summary.log", self.logs)
        log_filepath = self.get_file_path(test_config, "all.log") 
        with open(log_filepath, "w+") as log_output:
            for log_record in self.logs: 
                log_output.write(str(log_record) + "\n")
    
    def write_log_trace(self, test_config):
        self.write_analytics(test_config, "summary.log", self.logs[0])
        log_filepath = self.get_file_path(test_config, "all.log")
        with open(log_filepath, "w+") as log_output:
            for id in range(len(self.logs[0])): 
                log_output.write(str(self.logs[0][id]) + " " + str(self.logs[1][id]) + "\n")
        # 0: produced time; 1: processed time
    
    def check_performace(self, test_config):
        log_filepath = self.get_file_path(test_config, "all.log")
        with open(log_filepath, 'r') as log_output:
            lines = log_output.readlines()
            cnt = 0
            for line in lines:
                data = line.strip('\n').split(' ')
                if(float(data[-1]) <= self.slo):
                    cnt += 1
            total_cnt = len(lines)
        return total_cnt * 0.99 <= cnt, cnt, total_cnt, (self.slo)

    def random_test_period(self, test_config, time_len, rec = RecordingAction.NO_RECORD):
        start_time = time.time()
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        num = 0
        input = self.get_input(get_input_config_from_test_config(test_config))
        while(time.time() - start_time < time_len):
            num += 1
            if(test_config.nosync):
                start_time_sub = time.time()
                for i in range(test_config.batch_size):
                    self.inference(input, test_config.start, test_config.end, False, False)
                torch.cuda.synchronize()
                if(rec in RECORDING_ACTION_MAP["statistics"]):
                    self.logs.append((time.time() - start_time_sub) * 1000)
            else:
                self.inference(input, test_config.start, test_config.end, True, rec in RECORDING_ACTION_MAP["statistics"])
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)
    
    def trace_test_period(self, test_config, time_len, rec = RecordingAction.NO_RECORD):
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            pids = self.start_dcgm(test_config)
        self.logs = [[], []]
        test_start_time = time.time()
        time_limit = test_start_time + time_len
        cnt = 0
        input = self.get_input(get_input_config_from_test_config(test_config))
        while(True):
            req_time = self.trace.next()
            sleep_time = req_time + test_start_time - time.time()
            if(sleep_time > 0):
                time.sleep(sleep_time)
            if(time.time() > time_limit):
                break
            req_start_time = time.time()
            self.inference(input, test_config.start, test_config.end, True, rec)
            torch.cuda.synchronize()
            req_process_time = time.time() - req_start_time
            req_produced_time = time.time() - req_time - test_start_time
            if(RECORDING_ACTION_MAP["statistics"]):
                self.logs[0].append(req_process_time * 1000)
                # self.logs[1].append(req_produced_time * 1000)
                self.logs[1].append(req_produced_time * 1000)
            cnt += 1
        print("Processed: ", cnt)
        if(rec in RECORDING_ACTION_MAP["dcgm"]):
            stop_dcgm(pids)
    
    def random_test(self, test_config):
        self.random_test_period(test_config, self.warmup_time)
        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        recording_action = RecordingAction.ALL if test_config.recording_action == None else test_config.recording_action
        if(test_config.trace_path != None):
            self.trace_test_period(test_config, test_time, recording_action)
        else:
            self.random_test_period(test_config, test_time, recording_action)
        self.random_test_period(test_config, self.warmup_time)
        if(recording_action in RECORDING_ACTION_MAP["statistics"]):
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
        self.partial_model = {}
        self.partial_input = {}
        for batch_size in self.supported_batchsize:
            self.partial_input[batch_size] = {}
            self.partial_input[batch_size][0] = self.random_input[batch_size]
        for i in range(0, self.total_module, PARTIAL_MODEL_STRIDE):
            right = min(i+PARTIAL_MODEL_STRIDE, self.total_module)
            self.partial_model[i] = torch.nn.Sequential(*self.submodules[i:right]).cuda(self.device)
            if(right == self.total_module):
                break
            for batch_size in self.supported_batchsize:
                partial_input_current = self.partial_input[batch_size][i]
                self.partial_input[batch_size][right] = self._inference(partial_input_current, i, right)
    
    def warmup(self):
        warmup_config = TestConfig(1, None, 1, 0, self.total_module, -1, -1)
        warmup_config.fit(self)
        self.random_test_period(warmup_config, 5)
    
    def schedule(self, current_intensity):
        pass
    
    def execute_schedule_plan(self, schedule_plan):
        # Currently schedule plan is simple batch size
        input = self.get_input(TestConfig())
        pass

    def opportunistic_serve(self, test_config):
        test_time = self.test_time if test_config.test_time == None else test_config.test_time
        start_time = time.time()
        time_limit = start_time + test_time
        SCHEDULE_INTERVAL = 0.25
        while(True):
            if(time.time() > time_limit):
                break
            if(estimated_time < SCHEDULE_INTERVAL):
                time.sleep(estimated_time)
                current_intensity = read_intensity_from_pipe()
                schedule_plan = self.schedule(current_intensity)
                self.execute_schedule_plan(schedule_plan)
            time.sleep(SCHEDULE_INTERVAL)
            estimated_time = max(0, estimated_time - SCHEDULE_INTERVAL)

    def run(self) -> None:
        self.init_start()
        self.common_start()
        self.warmup()
        self.barrier.wait()
        while(True):
            action = self.pipe.recv()
            if(action == "serve"):
                self.serve()
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
            elif(action == "stop"):
                break
            else:
                raise NotImplementedError
        self.barrier.wait()
        self.shutdown()
    