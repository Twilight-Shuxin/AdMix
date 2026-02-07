import os
import time
import asyncio
from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, unpack_data_bytes
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.monitor import get_llm_intensity_unix
from profiling.constants import list_to_string
from pydantic import BaseModel

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
        return get_data(AIMDSchedulingDataFormat.ObservationData, client_socket)

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

class AIMDScheduler(Scheduler):
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

        if (uncompleted_tokens > 10 and self.profile[self.profile_pointer][1] * 1.1 < uncompleted_tokens):
            self.batch_size //= self.beta
        elif (self.last != self.profile_pointer and self.batch_size < self.max_batch_size):
            if(self.batch_size == 0):
                self.batch_size = 1
            else:
                self.batch_size += self.alpha
                self.batch_size = min(self.max_batch_size, self.batch_size)
        
        if(self.last != self.profile_pointer):
            self.last = self.profile_pointer
        return BasicSchedulePlan(self.batch_size)
    
    def _serve(self):
        self.profile = []
        self.profile_pointer = 0

        with open(os.path.join(self.model_worker.profile_path, self.model_worker.test_config.llm_tag + ".log"), "r") as profile_file:
            lines = profile_file.readlines()
            for line in lines:
                self.profile.append([float(x) for x in line.strip().split(" ")])
        
        self.batch_size = 1
        self.max_batch_size = self.model_worker.supported_batchsize[-1]
        self.alpha = self.model_worker.alpha
        self.beta = self.model_worker.beta
        self.last = -1
        self.logs = []

        self.model_worker.double_barrier.wait()
        self.model_worker.runtime_data.start_time_pointer = time.perf_counter()
        self.model_worker.runtime_data.time_limit = self.model_worker.runtime_data.start_time_pointer + self.model_worker.test_config.test_time
        
        time.sleep(0.1)
        while(time.perf_counter() < self.model_worker.runtime_data.time_limit):
            schedule_plan = self.schedule(get_llm_intensity_unix(self.data_format))
            schedule_plan.execute(self.model_worker)
        
