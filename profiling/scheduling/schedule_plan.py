import time
from profiling.constants import InputConfig
from plotting.constants import *

class SchedulePlan():
    def execute(self, model_worker, rec):
        pass
    
    def recorder(self, model_worker):
        pass

class BasicSchedulePlan(SchedulePlan):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def execute(self, model_worker):
        if(self.batch_size == 0):
            return
        input = model_worker.get_input(InputConfig(batch_size = self.batch_size))
        req_start_time = time.perf_counter()
        model_worker.inference(input)
        req_process_time = time.perf_counter() - req_start_time
        if(model_worker.test_config.record_statistics):
            model_worker.profile_data.logs.append((model_worker.test_config.batch_size, req_process_time * 1000))
            model_worker.profile_data.absolute_throughput += model_worker.test_config.batch_size
            model_worker.profile_data.absolute_throughput_time_bins[int((time.perf_counter() - model_worker.runtime_data.start_time_pointer) / PLOTTING_BIN_SIZE)] += model_worker.test_config.batch_size
            model_worker.profile_data.throughput += 1