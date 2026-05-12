import time
from admix.profiling.constants import InputConfig
from admix.plotting.constants import *

class SchedulePlan():
    def execute(self, model_worker):
        pass
    
    def recorder(self, model_worker):
        pass

class BasicSchedulePlan(SchedulePlan):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def execute(self, model_worker):
        if(self.batch_size == 0):
            return
        modul_input = model_worker.get_input(InputConfig(batch_size = self.batch_size))
        req_start_time = time.perf_counter()
        model_worker.inference(modul_input)
        req_process_time = time.perf_counter() - req_start_time
        model_worker.runtime_data.absolute_throughput += self.batch_size
        if(model_worker.test_config.recording_action.record_statistics):
            model_worker.profile_data.logs.append((self.batch_size, req_process_time * 1000))
            model_worker.profile_data.absolute_throughput_records.add(
                time.perf_counter() - model_worker.runtime_data.start_time_pointer, self.batch_size)

class ContextSchedulePlan(SchedulePlan):
    def __init__(self, mps, batch_size, stop = False):
        self.mps = mps
        self.batch_size = batch_size
        self.stop = stop

    def execute(self, model_worker):
        if self.stop:
            model_worker.context_teardown()
            return
        # context_switch is a no-op if already on this mps; otherwise switches context.
        model_worker.context_switch(self.mps)
        # Instruct the thread to loop inference continuously until the next plan arrives.
        model_worker.start_inference_loop(self.mps, self.batch_size)
