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

class ConstantSchedulingDataFormat(SchedulingDataFormat):
    class ObservationData(BaseModel):
        foo: int = 0

    def default_data():
        return ConstantSchedulingDataFormat.ObservationData()
    
    def pack_data(llm_worker):
        data =  ConstantSchedulingDataFormat.ObservationData(
            foo = 0)
        return pack_data_bytes(data)

    def read_data(client_socket):
        length = get_data_length(client_socket)
        return unpack_data_bytes(ConstantSchedulingDataFormat.ObservationData, client_socket, length)

class ConstantScheduler(Scheduler):
    data_format = ConstantSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self, batch_size = 16):
        self.batch_size = batch_size

    def schedule(self, intensity_data):
        return BasicSchedulePlan(self.batch_size)
    
    def serve(self, model_worker, test_time, rec):
        self.max_batch_size = model_worker.supported_batchsize[-1]
        model_worker.single_barrier.wait()
        model_worker.start_time_pointer = time.perf_counter()
        self.start_time_pointer = time.perf_counter()
        time_limit = self.start_time_pointer + test_time
        time.sleep(0.1)
        while(True):
            if(time.perf_counter() > time_limit):
                break
            schedule_plan = self.schedule(ConstantSchedulingDataFormat.ObservationData())
            schedule_plan.execute(model_worker, rec)
        
