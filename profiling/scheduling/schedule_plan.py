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

    # entering basic recorder: in statistics recording mode
    def recorder(self, model_worker):
        model_worker.absolute_throughput += self.batch_size
        model_worker.absolute_throughput_time_bins[int((time.perf_counter() - model_worker.start_time_pointer) / PLOTTING_BIN_SIZE)] += self.batch_size
        model_worker.throughput += 1

    def execute(self, model_worker, rec):
        # Currently schedule plan is simple batch size
        if(self.batch_size == 0):
            return
        if(rec):
            self.recorder(model_worker)
        input = model_worker.get_input(InputConfig(self.batch_size))
        model_worker.inference(input, 0, model_worker.total_module, True, rec)