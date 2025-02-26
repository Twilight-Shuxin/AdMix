from torch.multiprocessing import Process
from profiling.constants import *
import torch
import time
import numpy as np
import os

class ModelWorker(Process):
    def __init__(self, model_name, device, pipe, log_path):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.pipe = pipe
        self.log_path = log_path
        self.logs = []
        self.submodel = None
        self.supported_batchsize = [1, 2, 4, 8, 16, 32]
        self.dirname, filename = os.path.split(os.path.abspath(__file__))

    def serve():
       raise NotImplementedError
    
    def test():
        raise NotImplementedError
    
    def get_dcgm_logprefix(self, test_config):
        return os.path.join(self.log_path, "_".join([self.model_name, str(test_config.batch_size)]))

    def start_dcgm(self, dcgm_group, log_filepath_prefix):
        if(dcgm_group == -1):
            return []
        pids = []
        for profile_statistic in profile_group_dict[dcgm_group]:
            profile_id = profile_dict[profile_statistic]
            command = "{}/dcgm_monitor.sh {} {}".format(self.dirname, str(profile_id), 
                "_".join([log_filepath_prefix, str(profile_statistic)]) + ".log")
            os.system(command)
            with open('tmp.txt', 'r') as pid_file:
                pid = int(pid_file.read())
            pids.append(pid)
        return pids

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
    
    def write_log(self, test_config):
        tag = str(test_config.batch_size)
        if(test_config.nosync):
            tag += "_nosync"
        if(test_config.inter_flag):
            tag += "[" + str(test_config.start) + "-" + str(test_config.end) + "]"
        log_filepath = os.path.join(self.log_path, "_".join([self.model_name, tag, "summary.log"]))
        with open(log_filepath, "w+") as log_output:
            log_output.write("mean " + str(np.average(self.logs)) + "\n")
            log_output.write("min " + str(np.amin(self.logs)) + "\n")
            log_output.write("max " + str(np.amax(self.logs)) + "\n")
            log_output.write("var " + str(np.var(self.logs)) + "\n")
            log_output.write("90% " + str(np.percentile(self.logs, 90)) + "\n")
            log_output.write("95% " + str(np.percentile(self.logs, 95)) + "\n")
            log_output.write("99% " + str(np.percentile(self.logs, 99)) + "\n")
        log_filepath = os.path.join(self.log_path, "_".join([self.model_name, tag, "all.log"]))  
        with open(log_filepath, "w+") as log_output:
            for log_record in self.logs: 
                log_output.write(str(log_record) + "\n")

    def random_test_period(self, test_config, time_len, rec = False):
        start_time = time.time()
        if(rec):
            pids = self.start_dcgm(test_config.dcgm_group, self.get_dcgm_logprefix(test_config))
        print(test_config.batch_size, test_config.inter_flag)
        num = 0
        while(time.time() - start_time < time_len):
            input = self.get_input(test_config)
            num += 1
            if(test_config.nosync):
                start_time_sub = time.time()
                for i in range(test_config.batch_size):
                    self.inference(input, test_config.start, test_config.end, False, False)
                if(rec):
                    self.logs.append((time.time() - start_time_sub) * 1000)
                torch.cuda.synchronize()
            else:
                self.inference(input, test_config.start, test_config.end, True, rec)
        if(rec):
            stop_dcgm(pids)
    
    def random_test(self, test_config):
        self.random_test_period(test_config, 10)
        self.random_test_period(test_config, 40, True)
        self.random_test_period(test_config, 5)
        self.write_log(test_config)
    
    def common_start(self):
        self.submodules = self.model.get_submodules()
        self.total_module = len(self.submodules)
        self.partial_input = {}
        for batch_size in self.supported_batchsize:
            input = [self.random_input[batch_size]]
            for i in range(1, self.total_module):
                input.append(self._inference((self.random_input[batch_size],), 0, i))
            self.partial_input[batch_size] = input

    def run(self) -> None:
        self.init_start()
        self.common_start()
        (action) = self.pipe.recv()
        print("here", action)
        if(action == "serve"):
            self.serve()
        elif(action == "test"):
            test_config = self.pipe.recv()
            test_config.fit(self)
            self.random_test(test_config)
        else:
            raise NotImplementedError
    