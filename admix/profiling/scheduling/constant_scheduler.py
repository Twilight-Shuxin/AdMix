import time
from pydantic import BaseModel
from admix.profiling.models.workers.constants import SyncType
from admix.profiling.scheduling.schedule_plan import BasicSchedulePlan
from admix.profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, conn_get_data

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
        return conn_get_data(ConstantSchedulingDataFormat.ObservationData, client_socket)

class ConstantScheduler(Scheduler):
    data_format = ConstantSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self, batch_size = 16):
        self.batch_size = batch_size

    def schedule(self, intensity_data):
        return BasicSchedulePlan(self.batch_size)
    
    def _serve(self):
        self.start_serve()
        time.sleep(0.1)
        while(True):
            if(time.perf_counter() > self.model_client.runtime_data.time_limit):
                break
            schedule_plan = self.schedule(ConstantSchedulingDataFormat.ObservationData())
            self.model_client.broadcast_workers(schedule_plan, sync = SyncType.ALL)
        
