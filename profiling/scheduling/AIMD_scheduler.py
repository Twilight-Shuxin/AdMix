from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, unpack_data_bytes, get_data_length
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.monitor import get_llm_intensity_unix
from pydantic import BaseModel
from plotting.constants import *
from profiling.constants import EPS, list_to_string
from threading import Thread
import struct
import time
import signal
import os
import asyncio

class AIMDSchedulingDataFormat(SchedulingDataFormat):
    class ObservationData(BaseModel):
        current_timestamp: float = 0
        uncompleted_tokens: int = 0

    def default_data():
        return AIMDSchedulingDataFormat.ObservationData()
    
    def pack_data(llm_worker):
        data =  AIMDSchedulingDataFormat.ObservationData(
            current_timestamp=llm_worker.uncompleted_tokens_last_record[0], 
            uncompleted_tokens=llm_worker.uncompleted_tokens_last_record[1])
        return pack_data_bytes(data)

    def read_data(client_socket):
        length = get_data_length(client_socket)
        return unpack_data_bytes(AIMDSchedulingDataFormat.ObservationData, client_socket, length)

async def AIMD_monitor(llm_worker):
    llm_worker.uncompleted_tokens_last_record = (0,0)
    timestamp = 0
    try:
        while(True):
            llm_worker.uncompleted_tokens_last_record = (timestamp, llm_worker.uncompleted_tokens)
            await asyncio.sleep(0.1)
            timestamp += 0.1
    except asyncio.CancelledError:
        raise

async def AIMD_profiler(llm_worker, test_config):
    logs = []
    timestamp = 0
    try:
        print("started profiling")
        while(True):
            logs.append([timestamp, llm_worker.uncompleted_tokens])
            await asyncio.sleep(0.1)
            timestamp += 0.1
    except asyncio.CancelledError:
        file_path = os.path.join(llm_worker.profile_path, "-".join([llm_worker.model_name, test_config.tag])  + ".log")
        with open(file_path, "w") as profile_file:
            print("Writing logs to:", file_path)
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

    def __init__(self):
        self.batch_size = 1
        self.key = True
        self.last = -1
        self.logs = []

    def schedule(self, intensity_data):
        # intensity of queries waiting for first token
        current_timestamp = intensity_data.current_timestamp
        uncompleted_tokens = intensity_data.uncompleted_tokens
        while (self.profile_pointer + 1 < len(self.profile) and current_timestamp - self.profile[self.profile_pointer][0] > 0.025):
            self.profile_pointer += 1
        if (uncompleted_tokens > 10 and self.profile[self.profile_pointer][1] * 1.1 < uncompleted_tokens):
            self.batch_size //= self.beta
        elif (self.last != self.profile_pointer and self.batch_size < self.max_batch_size):
            if(self.batch_size == 0):
                self.batch_size = 1
            else:
                self.batch_size += self.alpha if self.batch_size != 1 else self.alpha - 1
            self.batch_size = min(self.max_batch_size, self.batch_size)
        if(self.last != self.profile_pointer):
            # print(current_timestamp, self.profile[self.profile_pointer], 
            #     uncompleted_tokens, self.batch_size)
            # self.logs.append([current_timestamp, self.profile[self.profile_pointer], uncompleted_tokens, self.batch_size])
            self.last = self.profile_pointer
        return BasicSchedulePlan(self.batch_size)
    
    def serve(self, model_worker, test_time, rec):
        self.profile = []
        self.profile_pointer = 0
        self.max_batch_size = model_worker.supported_batchsize[-1]
        self.alpha = model_worker.alpha
        self.beta = model_worker.beta
        with open(os.path.join(model_worker.profile_path, "-".join(["llm", self.llm_tag]) + ".log"), "r") as profile_file:
            lines = profile_file.readlines()
            for line in lines:
                self.profile.append([float(x) for x in line.strip().split(" ")])

        model_worker.single_barrier.wait()
        model_worker.start_time_pointer = time.perf_counter()
        self.start_time_pointer = time.perf_counter()
        time_limit = self.start_time_pointer + test_time
        time.sleep(0.1)
        while(True):
            if(time.perf_counter() > time_limit):
                break
            schedule_plan = self.schedule(get_llm_intensity_unix(AIMDScheduler.data_format))
            schedule_plan.execute(model_worker, rec)
        # for log in self.logs:
        #     print(log)
        
