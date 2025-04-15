from torch.multiprocessing import Process
from profiling.constants import *
import torch
import time
import numpy as np
from trace.trace_utils import Trace
import os

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
        self.submodel = None
        self.supported_batchsize = [1, 2, 4, 8, 16, 32]
        self.dirname, filename = os.path.split(os.path.abspath(__file__))

    def serve():
       raise NotImplementedError
    
    def test():
        raise NotImplementedError
    
    def get_trace(self, trace_path):
        self.trace = Trace(trace_path)

    def start_dcgm(self, test_config):
        if(test_config.dcgm_group == -1):
            return []
        pids = []
        for profile_statistic in profile_group_dict[test_config.dcgm_group]:
            profile_id = profile_dict[profile_statistic]
            log_filepath = os.path.join(self.log_path, "-".join([test_config.tag, str(profile_statistic)]) + ".log")
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

    def get_input(self, test_config):
        batch_size = test_config.batch_size
        if(test_config.nosync):
            batch_size = 1
        if test_config.inter_flag:
            return self.partial_input[batch_size][test_config.start],  
        else:
            return self.random_input[batch_size],
    
    def _inference(self, input, start, end):
        if(start != 0 or end != self.total_module): 
            if(self.submodel is None):
                result = torch.nn.Sequential(*self.submodules[start:end])(*input) 
            else:
                result = self.submodel(*input)
        else:
            result = self.model(*input)
        return result

    def inference(self, input, start, end, sync = False, rec = False):
        start_time = time.time()
        self._inference(input, start, end)
        if(sync): 
            torch.cuda.synchronize()
            if(rec):
                self.logs.append((time.time() - start_time) * 1000)
    
    def write_analytics(self, file_name, record_list):
        log_filepath = os.path.join(self.log_path, file_name)
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
        self.write_analytics("-".join([test_config.tag, "summary.log"]), self.logs)
        log_filepath = os.path.join(self.log_path, "-".join([test_config.tag, "all.log"]))  
        with open(log_filepath, "w+") as log_output:
            for log_record in self.logs: 
                log_output.write(str(log_record) + "\n")
    
    def write_log_trace(self, test_config):
        self.write_analytics("-".join([test_config.tag, "summary.log"]), self.logs[0])
        log_filepath = os.path.join(self.log_path, "-".join([test_config.tag, "all.log"]))  
        with open(log_filepath, "w+") as log_output:
            for id in range(len(self.logs[0])): 
                log_output.write(str(self.logs[0][id]) + " " + str(self.logs[1][id]) + "\n")

    def random_test_period(self, test_config, time_len, rec = False):
        start_time = time.time()
        if(rec):
            pids = self.start_dcgm(test_config)
        num = 0
        while(time.time() - start_time < time_len):
            input = self.get_input(test_config)
            num += 1
            if(test_config.nosync):
                start_time_sub = time.time()
                for i in range(test_config.batch_size):
                    self.inference(input, test_config.start, test_config.end, False, False)
                torch.cuda.synchronize()
                if(rec):
                    self.logs.append((time.time() - start_time_sub) * 1000)
            else:
                self.inference(input, test_config.start, test_config.end, True, rec)
        if(rec):
            stop_dcgm(pids)
    
    def trace_test_period(self, test_config, time_len, rec = False):
        if(rec):
            pids = self.start_dcgm(test_config)
        self.logs = [[], []]
        test_start_time = time.time()
        time_limit = test_start_time + time_len
        cnt = 0
        while(True):
            req_time = self.trace.next()
            sleep_time = req_time + test_start_time - time.time()
            if(sleep_time > 0):
                time.sleep(sleep_time)
            if(time.time() > time_limit):
                break
            input = self.get_input(test_config)
            req_start_time = time.time()
            self.inference(input, test_config.start, test_config.end, True, rec)
            torch.cuda.synchronize()
            req_process_time = time.time() - req_start_time
            req_produced_time = time.time() - req_time - test_start_time
            if(rec):
                self.logs[0].append(req_process_time * 1000)
                # self.logs[1].append(req_produced_time * 1000)
                self.logs[1].append(req_produced_time * 1000)
            cnt += 1
        print("Processed: ", cnt)
        if(rec):
            stop_dcgm(pids)
    
    def random_test(self, test_config):
        self.random_test_period(test_config, 10)
        if(test_config.trace_path != None):
            self.trace_test_period(test_config, 40, True)
        else:
            self.random_test_period(test_config, 40, True)
        self.random_test_period(test_config, 5)
        if(test_config.trace_path != None):
            self.write_log_trace(test_config)
        else:
            self.write_log_normal(test_config)
    
    def common_start(self):
        self.submodules = self.model.get_submodules()
        self.total_module = len(self.submodules)
        # self.partial_input = {}
        # for batch_size in self.supported_batchsize:
        #     input = [self.random_input[batch_size]]
        #     for i in range(1, self.total_module):
        #         input.append(self._inference((self.random_input[batch_size],), 0, i))
        #     self.partial_input[batch_size] = input
    
    def warmup(self):
        warmup_config = TestConfig(1, None, 1, 0, self.total_module, -1, -1)
        warmup_config.fit(self)
        self.random_test_period(warmup_config, 5)

    def run(self) -> None:
        self.init_start()
        self.common_start()
        self.warmup()
        self.barrier.wait()
        while(True):
            (action) = self.pipe.recv()
            if(action == "serve"):
                self.serve()
            elif(action == "test"):
                test_config = self.pipe.recv()
                test_config.fit(self)
                print("here", test_config.tag)
                self.random_test(test_config)
            elif(action == "tag"):
                test_config = self.pipe.recv()
                self.pipe.send(self.get_tag(test_config))
            elif(action == "stop"):
                break
            else:
                raise NotImplementedError
    