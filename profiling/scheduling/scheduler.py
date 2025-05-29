from collections import namedtuple
from profiling.constants import InputConfig
from plotting.constants import *
import struct
import time

BasicSchedulePlan = namedtuple('BasicSchedulePlan', ['batch_size'])
SchedulerInfo = namedtuple('SchedulerInfo', ['scheduler', 'default_data', 'pack_func', 'schedule_plan'])

# entering basic recorder: in statistics recording mode
def basic_recorder(model_worker, schedule_plan : BasicSchedulePlan):
    current_timestamp = time.perf_counter() - model_worker.start_time_pointer
    model_worker.absolute_throughput += schedule_plan.batch_size
    model_worker.throughput += 1
    model_worker.absolute_throughput_time_bins[int(current_timestamp / BIN_SIZE)] += schedule_plan.batch_size

def BasicSchedulePlanExecute(model_worker, schedule_plan, rec):
    # Currently schedule plan is simple batch size
    if(schedule_plan.batch_size == 0):
        return
    if(rec):
        basic_recorder(model_worker, schedule_plan)
    input = model_worker.get_input(InputConfig(schedule_plan.batch_size))
    model_worker.inference(input, 0, model_worker.total_module, True, rec)

schedule_plan_executors = {
    'BasicSchedulePlan': BasicSchedulePlanExecute,
}

class Scheduler():
    def schedule(self, intensity_data):
        return BasicSchedulePlan(2)

def default_AIMDScheduler_data():
    return 0,0,0

def pack_AIMDScheduler_data(llm_worker):
        return struct.pack("iii", llm_worker.queue_record[0], llm_worker.record_throughput, llm_worker.record_intensity)

class AIMDScheduler(Scheduler):
    def __init__(self):
        self.batch_size = 0
        self.max_batch_size = 32
        self.pack_func = pack_AIMDScheduler_data

    def schedule(self, intensity_data):
        # intensity of queries waiting for first token
        current_intensity, record_throughput, record_intensity = intensity_data
        if(current_intensity > 30):
            time.sleep(0.5)
            return BasicSchedulePlan(0)
        # return BasicSchedulePlan(1)
        return BasicSchedulePlan(2)

schedulers = {
    "AIMD": SchedulerInfo(AIMDScheduler, default_AIMDScheduler_data(), pack_AIMDScheduler_data, 'BasicSchedulePlan'),
}