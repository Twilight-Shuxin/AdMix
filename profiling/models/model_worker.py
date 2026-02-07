import torch
from torch.multiprocessing import Process
import time
import inspect
import numpy as np
from multiprocessing import Process as StdProcess, Pipe
import os
import csv
import json 
from profiling.constants import *
from trace.trace_utils import SimpleTimeTrace
from profiling.scheduling.utils import get_scheduler
from plotting.dcgm_plot import plot_dcgm


# device is limited to CUDA_VISIBLE_DEVICE set before worker creation, thus all worker only see 'cuda'
class ModelWorker(Process):
    def __init__(self, model_name, device_id, mps, pipe, barrier, log_path, profile_path):
        self.model_name = model_name
        self.mps = mps
        self.pipe = pipe
        self.double_barrier, self.barrier = barrier
        self.log_path = log_path
        self.profile_path = profile_path
        self.device = "cuda"
        self.true_device_id = int(device_id) if device_id.isdigit() else device_id
        self.supported_batchsize = [x for x in range(1,33)]
        self.dirname, filename = os.path.split(os.path.abspath(__file__))
        self.beta = 2
        super().__init__()
    
    def init_start(self):
        return
    
    def shutdown(self):
        return
    
    """
        Setup the parameters related to submodules.
        FIXME: Currently ignored all partial model setup due to memory restrictions.
    """
    def submodules_setup(self):
        self.submodules = None
        self.total_module = -1
        if hasattr(self, "model") and hasattr(self.model, "get_submodules") and callable(self.model.get_submodules):
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
    
    @classmethod
    def get_input_config_from_test_config(cls, test_config):
        return InputConfig(batch_size = test_config.batch_size, nosync = test_config.nosync, start = test_config.start, 
            inter_flag = test_config.inter_flag) 
    
    def get_input(self, input_config):
        batch_size = input_config.batch_size
        if(input_config.nosync):
            batch_size = 1
        if input_config.inter_flag:
            return self.partial_input[batch_size][input_config.start]
        else:
            return self.random_input[batch_size]
    
    # Setup submodel under current test_configuration: self.submodel = model[start:end]
    def setup_test_submodel(self):
        if(self.model_name != "bert" and self.test_config.inter_flag):
            self.submodel = torch.nn.Sequential(*self.submodules[self.test_config.start : self.test_config.end])
    
    # Get the trace according to trace path
    def get_trace(self):
        if(not self.test_config.trace_path):
            return
        self.test_config.trace = SimpleTimeTrace(self.test_config.trace_path)
    
    """
        Get the tag of oneself in the format of:
            model_name(bs_m70)[1,2]
            model_name(bs)[1,2]
            model_name(bs)
            model_name(m50)
            model_name
    """
    def get_self_tag(self, test_config):
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
        if(test_config.inter_flag):
            sub_tag += "[" + str(test_config.start) + "," + str(test_config.end) + "]"
        return tag + sub_tag
    
    """
        Get the full tag(self tag + global tag) for given test_configuration
        If model number is greater than 1, than self tag is reduced to model name because full name is included in global tag;
        yet if model number is 1 / no global tag available, than simply use full self tag
    """
    def get_full_tag(self, test_config):
        if(test_config.global_tag != None and test_config.global_cnt > 1):
            return "-".join([self.model_name, test_config.global_tag])
        else:
            return self.get_self_tag(test_config)
    
    def init_profile_data(self):
        self.profile_data = ProfileData()
        if(self.test_config.record_statistics):
            self.profile_data.logs = []
            self.profile_data.absolute_throughput_records = TimeBinRecord(init = 0, time_limit = self.test_config.test_time)
    
    def init_runtime_data(self):
        self.runtime_data = RuntimeData()
        self.runtime_data.absolute_throughput = 0
    
    def init_scheduler_data(self):
        self.scheduler_data = SchedulerData()
        self.scheduler_data.scheduler = get_scheduler(self.test_config.scheduler)

    def setup_testenv(self, test_config):
        self.test_config = test_config
        
        self.setup_test_submodel()
        self.get_trace()
        self.input_config = self.get_input_config_from_test_config(self.test_config)
        self.test_config.full_tag = self.get_full_tag(self.test_config)
    
    def init_testenv_data(self):
        self.init_profile_data()
        self.init_runtime_data()
        self.init_scheduler_data()

    def get_dcgm_file_path(self, profile_statistic):
        return os.path.join(self.log_path, "-".join([self.test_config.full_tag, str(profile_statistic)]) + ".log")

    def start_dcgm(self):
        if(self.test_config.dcgm_group == -1):
            return None
        dcgm_procs = []
        for profile_statistic in profile_group_dict[self.test_config.dcgm_group]:
            profile_id = profile_dict[profile_statistic]
            log_filepath = self.get_dcgm_file_path(profile_statistic)
            command = "{}/start_dcgm_dmon.sh {} {} \"{}\"".format(self.dirname, str(self.true_device_id + 2), str(profile_id), 
                log_filepath)
            os.system(command)
            with open('dcgm_pid.log', 'r') as pid_file:
                pid = int(pid_file.read())
            dcgm_procs.append((pid, command))
        parent_conn, child_conn = Pipe()
        monitor_proc = StdProcess(target=dcgm_dmon_monitor, args=(dcgm_procs, child_conn))
        monitor_proc.start()
        return parent_conn
            
    def inference(self, input, start = None, end = None, sync = True):
        if((start is not None) and (end is not None) and (start != 0 or end != self.total_module)): 
            result = input
            for i in range(start, end, PARTIAL_MODEL_STRIDE):
                result = self.partial_model[i](result)
        else:
            result = self.model(input)
        if(sync): 
            torch.cuda.synchronize()
        return result
            
    # Get the filepath as: self.log_path/full_tag-filename; or self.log_path/subpath/full_tag-filename if subpath specified.
    def get_file_path(self, file_name, subpath = None):
        if(subpath):
            return os.path.join(self.log_path, subpath, "-".join([self.test_config.full_tag, file_name]))
        return os.path.join(self.log_path, "-".join([self.test_config.full_tag, file_name]))
    
    def write_analytics(self, file_name, record_list):
        log_filepath = self.get_file_path(file_name)
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
    
    def write_all(self, file_name, record_list):
        log_filepath = self.get_file_path(file_name)
        with open(log_filepath, "w+") as log_output:
            for log_record in record_list: 
                log_output.write(list_to_string(log_record) + "\n")
    
    def write_list_to_csv(self, file_name, headers, record_list, subpath = None):
        log_filepath = self.get_file_path(file_name, subpath)
        with open(log_filepath, "w", newline="", encoding="utf-8") as log_output:
            writer = csv.writer(log_output)
            writer.writerow(headers)
            for record in record_list:
                if(isinstance(record, list) or isinstance(record, tuple)):
                    writer.writerow(record)  # Writes entire list
                else:
                    writer.writerow([record])
    
    """
        Write logs:
            profile_data.logs: [(batch_size, latency)] -> analytics of latency & all
            profile_data.absolute_throughput_time_bins: throughput at each time step
    """
    def write_logs(self):
        self.write_analytics("summary.log", [x[1] for x in self.profile_data.logs])
        self.write_list_to_csv("all.csv", ['Batch Size', 'Latency (ms)'], self.profile_data.logs)
        self.write_all("flow.log", self.profile_data.absolute_throughput_records.append_timestamps())
  

    """
        Return performance metrics collected in the last test run.
    """
    def check_performace(self):
        return self.runtime_data.absolute_throughput
    
    """
        Perform inferences for GPU & model warmup.
    """
    def warmup(self):
        warmup_iter = 10
        model_input = self.get_input(self.input_config)
        for _ in range(warmup_iter):
            self.inference(model_input, self.test_config.start, self.test_config.end, sync = True)
    
    """
        Perform test according to self.test_config.
        If scheduler is set, use scheduler's serve method instead.
        All runtime settings have been completed by the wrapper function 'test' ↓
    """
    def _test(self):
        if(self.scheduler_data.scheduler):
            self.scheduler_data.scheduler.serve(self)
            return

        # Start test
        self.runtime_data.start_time_pointer = time.perf_counter()
        self.runtime_data.time_limit = self.runtime_data.start_time_pointer + self.test_config.test_time
        model_input = self.get_input(self.input_config)
        while(time.perf_counter() < self.runtime_data.time_limit):
            if(self.test_config.trace_path):
                req_time = self.test_config.trace.next()
                sleep_time = req_time + self.runtime_data.start_time_pointer - time.perf_counter()
                if(sleep_time > 0):
                    time.sleep(sleep_time)
                if(time.perf_counter() > self.runtime_data.time_limit):
                    break

            req_start_time = time.perf_counter()
            if(self.test_config.nosync):
                for i in range(self.test_config.batch_size):
                    self.inference(model_input, self.test_config.start, self.test_config.end, sync=False)
                torch.cuda.synchronize()
            else:
                self.inference(model_input, self.test_config.start, self.test_config.end, sync=True)
            req_process_time = time.perf_counter() - req_start_time

            self.runtime_data.absolute_throughput += self.test_config.batch_size
            self.profile_data.absolute_throughput_records.add(time.perf_counter() - self.runtime_data.start_time_pointer, self.test_config.batch_size)
            if(self.test_config.record_statistics):
                self.profile_data.logs.append((self.test_config.batch_size, req_process_time * 1000))
        # End test

    """
        Perform test according to self.test_config, supports sequential test, trace test and scheduler test. 
        Initializes the runtime settings for test and execute test.
    """
    def test(self):
        if(self.test_config.record_dcgm):
            dcgm_monitor_conn = self.start_dcgm()

        if inspect.iscoroutinefunction(self._test):
            self.loop.run_until_complete(self._test())
        else:
            self._test()
    
        if(self.test_config.record_dcgm):
            stop_dcgm(dcgm_monitor_conn)
        if(self.test_config.record_statistics):
            self.write_logs()
    
    """
        Collect the dcgm statistics recorded in files in the previous test runs.
        Plot certain specified dcgm statistics.
        FIXME: Current collected dcgm statistics are those in self.test_config, check if this is properly set.
    """
    def collect_dcgm(self, dcgm_groups):
        plot_dcgm_statistics = ["sm_active", "dram_active"]
        dcgm_data = {}
        for dcgm_group in dcgm_groups:
            if(dcgm_group == -1):
                continue
            for profile_statistic in profile_group_dict[dcgm_group]:
                dcgm_data_path = self.get_dcgm_file_path(profile_statistic)
                dcgm_data[profile_statistic] = get_dcgm_analysis(dcgm_data_path)
                if(profile_statistic in plot_dcgm_statistics):
                    records = get_dcgm_data(dcgm_data_path)
                    plot_dcgm(records, os.path.join(self.log_path, self.test_config.full_tag + "-"), profile_statistic)
        return dcgm_data if dcgm_data else None
    
    """
        Profile model to get the 99 percentile latency of each batch size setting.
        Save profiled results into self.profile_path/model_name.json.
    """
    def profile(self):
        profile_iter = 15
        warmup_iter = 5
        profile_data = {
            "model_name": self.model_name
        }
        for batch_size in self.supported_batchsize:
            spent_time = []
            for _ in range(warmup_iter):
                self.inference(self.random_input[batch_size], sync = True)
            for _ in range(profile_iter):
                start_time = time.perf_counter()
                self.inference(self.random_input[batch_size], sync = True)
                spent_time.append(time.perf_counter() - start_time)
            profile_data[batch_size] = np.percentile(spent_time, 99)
        with open(os.path.join(self.profile_path, self.model_name + ".json"), "w") as profile_file:
            json.dump(profile_data, profile_file, indent=4)
    
    """
        Read profiled data from self.profile_path/model_name.json.
    """
    def read_profiled_data(self):
        with open(os.path.join(self.profile_path, self.model_name + ".json"), "r") as profile_file:
            self.profiled_data = json.load(profile_file)
        return self.profiled_data

    def run(self) -> None:
        self.init_start()
        self.submodules_setup()
        self.barrier.wait()
        while(True):
            action = self.pipe.recv()
            if(action == "profile"):
                # FIXEME: Currently profile does not have test environment activated: this may lead 
                # to troubles if profile mode is used for LLM worker.
                self.profile()
                self.double_barrier.wait()
            elif(action == "test"):
                self.setup_testenv(self.pipe.recv())
                self.test_config.set_warmup()
                self.init_testenv_data()
                self.warmup()
                self.test_config.set_test()
                self.init_testenv_data()
                self.barrier.wait()
                self.test()
                self.barrier.wait()
            elif(action == "self_tag"):
                test_config = self.pipe.recv()
                self.pipe.send(self.get_self_tag(test_config))
            elif(action == "full_tag"):
                test_config = self.pipe.recv()
                self.pipe.send(self.get_full_tag(test_config))
            elif(action == "collect"):
                collect_action = self.pipe.recv()
                if(collect_action == "dcgm"):
                    dcgm_groups = self.pipe.recv()
                    self.pipe.send(self.collect_dcgm(dcgm_groups))
                self.pipe.send(self.check_performace())
            elif(action == "get_total_modules"):
                self.pipe.send(self.total_module)
            elif(action == "stop"):
                break
            else:
                raise NotImplementedError
        self.barrier.wait()
        self.shutdown()
    