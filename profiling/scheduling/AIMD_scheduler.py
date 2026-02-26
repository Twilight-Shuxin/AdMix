import os
import time
import asyncio
from pydantic import BaseModel
from profiling.constants import list_to_string
from profiling.models.workers.constants import SyncType
from profiling.scheduling.monitor import get_llm_intensity_unix
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.scheduler import AIMDBaseScheduler, SchedulingDataFormat, pack_data_bytes, conn_get_data

class AIMDSchedulingDataFormat(SchedulingDataFormat):
    class ObservationData(BaseModel):
        current_timestamp: float = 0
        uncompleted_tokens: int = 0

    @classmethod
    def default_data(cls):
        return AIMDSchedulingDataFormat.ObservationData()
    
    @classmethod
    def pack_data(cls, llm_worker):
        data =  AIMDSchedulingDataFormat.ObservationData(
            current_timestamp=llm_worker.scheduler_data.uncompleted_tokens_last_record[0], 
            uncompleted_tokens=llm_worker.scheduler_data.uncompleted_tokens_last_record[1])
        return pack_data_bytes(data)
    
    @classmethod
    def read_data(cls, client_socket):
        return conn_get_data(AIMDSchedulingDataFormat.ObservationData, client_socket)

async def AIMD_monitor(llm_worker):
    llm_worker.scheduler_data.uncompleted_tokens_last_record = (0,0)
    timestamp = 0
    try:
        while(True):
            llm_worker.scheduler_data.uncompleted_tokens_last_record = (timestamp, llm_worker.scheduler_data.uncompleted_tokens)
            await asyncio.sleep(0.1)
            timestamp += 0.1
    except asyncio.CancelledError:
        raise

async def AIMD_profiler(llm_worker):
    logs = []
    timestamp = 0
    try:
        print("Started scheduler profiling")
        while(True):
            logs.append([timestamp, llm_worker.scheduler_data.uncompleted_tokens])
            await asyncio.sleep(AIMDScheduler.scheduling_interval)
            timestamp += 0.1
    except asyncio.CancelledError:
        file_path = os.path.join(llm_worker.profile_path, llm_worker.test_config.full_tag  + ".log")
        with open(file_path, "w") as profile_file:
            print("Writing profiled results to:", file_path)
            for log_record in logs: 
                profile_file.write(list_to_string(log_record) + "\n")
        raise
    except Exception as e:
        print(e)

class AIMDScheduler(AIMDBaseScheduler):
    data_format = AIMDSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan
    profiling_executor = AIMD_profiler
    monitor = AIMD_monitor
    scheduling_interval = 0.1
    half_scheduling_interval = scheduling_interval / 2

    def schedule(self, intensity_data):
        current_timestamp = intensity_data.current_timestamp
        uncompleted_tokens = intensity_data.uncompleted_tokens

        while (self.profile_pointer + 1 < len(self.profile) and
             (current_timestamp - self.profile[self.profile_pointer][0] > self.half_scheduling_interval)):
            self.profile_pointer += 1

        action = self.Action.FLAT
        if (uncompleted_tokens > 10 and self.profile[self.profile_pointer][1] * 1.1 < uncompleted_tokens):
            action = self.Action.DOWN
        elif (self.last != self.profile_pointer and self.batch_size < self.max_batch_size):
            action = self.Action.UP
        self.execute_action(action)
        
        if(self.last != self.profile_pointer):
            self.last = self.profile_pointer
        return BasicSchedulePlan(self.batch_size)
    
    def _serve(self):
        self.profile = []
        self.profile_pointer = 0

        with open(os.path.join(self.model_client.profile_path, self.model_client.get_self_tag(self.model_client.test_config) + ".log"), "r") as profile_file:
            lines = profile_file.readlines()
            for line in lines:
                self.profile.append([float(x) for x in line.strip().split(" ")])
        
        self.batch_size = 1
        self.max_batch_size = self.model_client.supported_batchsize[-1]
        self.alpha = self.model_client.alpha
        self.beta = self.model_client.beta
        self.last = -1
        self.logs = []

        self.start_serve()
        
        time.sleep(0.1)
        while(time.perf_counter() < self.model_client.runtime_data.time_limit):
            schedule_plan = self.schedule(get_llm_intensity_unix(self.data_format))
            self.model_client.broadcast_workers(schedule_plan, sync = SyncType.ALL)
        
