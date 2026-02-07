from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, get_data
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from pydantic import BaseModel
import time

class ConstantSchedulingDataFormat(SchedulingDataFormat):
    class ObservationData(BaseModel):
        foo: int = 0

    @classmethod
    def default_data(cls):
        return ConstantSchedulingDataFormat.ObservationData()
    
    @classmethod
    def pack_data(cls, llm_worker):
        data = ConstantSchedulingDataFormat.ObservationData(foo = 0)
        return pack_data_bytes(data)

    @classmethod
    def read_data(cls, client_socket):
        return get_data(ConstantSchedulingDataFormat.ObservationData, client_socket)

class ConstantScheduler(Scheduler):
    data_format = ConstantSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self, batch_size = 16):
        self.batch_size = batch_size

    def schedule(self, intensity_data):
        return BasicSchedulePlan(self.batch_size)
    
    def serve(self):
        self.model_worker.double_barrier.wait()
        self.model_worker.runtime_data.start_time_pointer = time.perf_counter()
        self.model_worker.runtime_data.time_limit = self.model_worker.runtime_data.start_time_pointer + self.model_worker.test_config.test_time
        time.sleep(0.1)
        while(True):
            if(time.perf_counter() > self.model_worker.runtime_data.time_limit):
                break
            schedule_plan = self.schedule(ConstantSchedulingDataFormat.ObservationData())
            schedule_plan.execute(self.model_worker)
        
