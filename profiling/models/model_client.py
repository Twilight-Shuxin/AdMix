import os
import csv
import json 
import time
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing import Process as StdProcess, Pipe
from profiling.constants import *
from plotting.dcgm_plot import plot_dcgm
from profiling.models.utils import avg_dict
from profiling.scheduling.utils import get_scheduler
from profiling.models.workers.constants import SyncType
from profiling.scheduling.monitor import llm_server_rpc_retry

class ModelClient(Process):
    def __init__(self, model_name, devices, mps, pipe, barrier, log_path, profile_path):
        super().__init__()
        self.model_name = model_name
        self.mps = mps
        self.pipe = pipe
        self.double_barrier, self.barrier = barrier
        self.log_path = log_path
        self.profile_path = profile_path
        self.devices = devices
        self.dirname, filename = os.path.split(os.path.abspath(__file__))
        self.worker_pipes = []
        self.beta = 2
    
    # To be implemented by descendant classes
    def init_start(self):
        pass
    
    def start_worker(self, devices):
        device_init(self.devices, devices, self.mps)
        pipe_parent, pipe_child = mp.Pipe()
        worker = self.model_worker(self.supported_batchsize, pipe_child, self.worker_barrier, self.single_worker_barrier,
                                    self.log_path, self.profile_path,)
        worker.start()
        self.worker_pipes.append(pipe_parent)
    
    def start_workers(self):
        self.worker_barrier = mp.Barrier(len(self.devices) + 1)
        self.single_worker_barrier = mp.Barrier(2)
        for device in self.devices:
            self.start_worker([device])
        self.worker_barrier.wait()

    def broadcast_workers(self, data, sync = SyncType.NO_SYNC):
        for pipe in self.worker_pipes:
            pipe.send(data)
            pipe.send(sync)
        if(sync == SyncType.ALL):
            self.worker_barrier.wait()
    
    def command_worker(self, pipe, command, additional_data = [], sync = SyncType.NO_SYNC):
        pipe.send(command)
        pipe.send(sync)
        for data in additional_data:
            pipe.send(data)
        if(sync == SyncType.SINGLE):
            self.single_worker_barrier.wait()
    
    def command_workers(self, command, additional_data = [], sync = SyncType.NO_SYNC):
        for pipe in self.worker_pipes:
            self.command_worker(pipe, command, additional_data, sync = sync)
        if(sync == SyncType.ALL):
            self.worker_barrier.wait()
    
    """
        Get the tag of oneself in the format of:
            model_name(bs_m70)[1,2]
            model_name(bs)[1,2]
            model_name(bs)
            model_name(m50)
            model_name
    """
    def get_self_tag(self, test_config = None):
        if(not test_config):
            test_config = self.test_config
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
        yet if model number is 1 / no global tag available, than simply use full self tag + '-d{x}' (number of devices used)
    """
    def get_full_tag(self, test_config = None):
        if(not test_config):
            test_config = self.test_config
        if(test_config.global_tag != None and test_config.global_cnt > 1):
            return "-".join([self.model_name, test_config.global_tag])
        else:
            return self.get_self_tag(test_config) + f'-d{len(self.devices)}'
    
    """
        Read profiled data from self.profile_path/model_name.json.
    """
    def read_profiled_data(self):
        with open(os.path.join(self.profile_path, self.model_name + ".json"), "r") as profile_file:
            self.profiled_data = json.load(profile_file)
        return self.profiled_data
    
    def profile(self):
        self.command_workers("profile")
        worker_profile_datas = []
        for pipe in self.worker_pipes:
            worker_profile_datas.append(pipe.recv())
        profile_data = {"model_name": self.model_name}
        profile_data.update(avg_dict(worker_profile_datas))
        with open(os.path.join(self.profile_path, self.model_name + ".json"), "w") as profile_file:
            json.dump(profile_data, profile_file, indent=4)
    
    def setup_testenv(self, test_config):
        self.test_config = test_config    
        
        # NOTE: Currently, among all three tags only one is used by llm client's model worker:
        # self.test_config.self_full_tag is '{self tag}-dx', which is the profile name of llm worker's scheduler profile.
        self.test_config.self_tag = self.get_self_tag()
        self.test_config.self_full_tag = self.get_self_tag() + f'-d{len(self.devices)}'
        self.test_config.full_tag = self.get_full_tag()

        self.command_workers("setup_testenv", [test_config])
    
    def get_total_modules(self):
        self.command_worker(self.worker_pipes[0], "get_total_modules")
        return self.worker_pipes[0].recv()
    
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
    
    def write_csv(self, file_name, headers, record_list, subpath = None):
        write_list_to_csv(self.get_file_path(file_name, subpath), headers, record_list)
    
    """
        Write logs:
            profile_data.logs: [(batch_size, latency)] -> analytics of latency & all
            profile_data.absolute_throughput_time_bins: throughput at each time step
    """
    def write_logs(self):
        self.write_analytics("summary.log", [x[2] for x in self.profile_data.logs])
        self.write_csv("all.csv", ['Device_id', 'Batch Size', 'Latency (ms)'], self.profile_data.logs)
        self.write_all("flow.log", self.profile_data.absolute_throughput_records.append_timestamps())
    
    def collect_worker_data(self):
        self.profile_data = ProfileData()
        self.profile_data.logs = []

        self.runtime_data = RuntimeData()
        self.runtime_data.absolute_throughput = 0
        for idx, pipe in enumerate(self.worker_pipes):
            self.command_worker(pipe, "collect_performance_data")
            self.runtime_data.absolute_throughput += pipe.recv()

        if(self.test_config.record_statistics):
            self.profile_data.logs = []
            self.profile_data.absolute_throughput_records = TimeBinRecord(init = 0, time_limit = self.test_config.test_time)
            for idx, pipe in enumerate(self.worker_pipes):
                self.command_worker(pipe, "collect_profiled_data")
                logs = pipe.recv()
                self.profile_data.logs.extend([(self.devices[idx],) + log for log in logs])
                self.profile_data.absolute_throughput_records.update(pipe.recv(), "sum")
    
    """
        Return performance metrics collected in the last test run.
    """
    def check_performace(self):
        return self.runtime_data.absolute_throughput
    
    def get_dcgm_file_path(self, profile_statistic):
        return os.path.join(self.log_path, "-".join([self.test_config.full_tag, str(profile_statistic)]) + ".log")

    def start_dcgm(self):
        if(self.test_config.dcgm_group == -1):
            return None
        cmd = f"gmanager dcgm_addg tmp"
        p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate()
        group_id = int(stdout.strip())
        
        cmd = f"gmanager dcgm_add2g {group_id} {' '.join(list(map(str, self.devices)))}"
        os.system(cmd)

        dcgm_procs = []
        for profile_statistic in profile_group_dict[self.test_config.dcgm_group]:
            profile_id = profile_dict[profile_statistic]
            log_filepath = self.get_dcgm_file_path(profile_statistic)
            command = "{}/start_dcgm_dmon.sh {} {} \"{}\"".format(self.dirname, group_id, str(profile_id), 
                log_filepath)
            os.system(command)
            with open(DCGM_TMP_FILE, 'r') as pid_file:
                pid = int(pid_file.read())
            dcgm_procs.append((pid, command))

        parent_conn, child_conn = Pipe()
        monitor_proc = StdProcess(target=dcgm_dmon_monitor, args=(group_id, dcgm_procs, child_conn))
        monitor_proc.start()
        return parent_conn
    
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
    
    def init_runtime_data(self):
        self.runtime_data = RuntimeData()
    
    def init_scheduler_data(self):
        self.scheduler_data = SchedulerData()
        self.scheduler_data.scheduler = get_scheduler(self.test_config.scheduler)
    
    def init_testenv_data(self):
        self.init_runtime_data()
        self.init_scheduler_data()
    
    def _test(self):
        if(self.test_config.scheduler):
            self.test_config.test_time = llm_server_rpc_retry("test_time")
            # Updated test_config, so setup test environment again to ensure correctness.
            self.setup_testenv(self.test_config)
            self.init_testenv_data()
            self.scheduler_data.scheduler.serve(self)
            self.broadcast_workers(None, sync = SyncType.ALL)
        else:
            self.command_workers("test", sync = SyncType.ALL)
    
    def test(self):
        if(self.test_config.record_dcgm):
            dcgm_monitor_conn = self.start_dcgm()    
        self._test()
        if(self.test_config.record_dcgm):
            stop_dcgm(dcgm_monitor_conn)
        
        self.collect_worker_data()
        if(self.test_config.record_statistics):
            self.write_logs()

    def run(self) -> None:
        self.init_start()
        self.start_workers()
        self.barrier.wait()
        while(True):
            action = self.pipe.recv()
            match action:
                case "profile":
                    self.profile()
                    self.double_barrier.wait()
                case "test":
                    self.setup_testenv(self.pipe.recv())
                    self.command_workers("warmup", sync = SyncType.ALL)
                    self.barrier.wait()
                    self.test()
                    self.barrier.wait()
                case "self_tag":
                    test_config = self.pipe.recv()
                    self.pipe.send(self.get_self_tag(test_config))
                case "full_tag":
                    test_config = self.pipe.recv()
                    self.pipe.send(self.get_full_tag(test_config))
                case "collect":
                    collect_action = self.pipe.recv()
                    if(collect_action == "dcgm"):
                        dcgm_groups = self.pipe.recv()
                        self.pipe.send(self.collect_dcgm(dcgm_groups))
                    self.pipe.send(self.check_performace())
                case "get_total_modules":
                    self.pipe.send(self.get_total_modules())
                case "stop":
                    self.command_workers("stop")
                    break
                case _:
                    raise NotImplementedError
        self.barrier.wait()
    
  