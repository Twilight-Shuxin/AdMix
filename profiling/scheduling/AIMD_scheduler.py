from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.monitor import get_llm_intensity_unix
from plotting.constants import *
from profiling.constants import EPS
import struct
import time

class AIMDSchedulingDataFormat(SchedulingDataFormat):
    def default_data():
        return 0,0,0
    
    def pack_data(llm_worker):
        current_timestamp = (time.perf_counter() - llm_worker.start_time_pointer)
        current_bin_timestamp = int(current_timestamp / PLOTTING_BIN_SIZE)
        left = max(current_bin_timestamp - 5, 0)
        ft = 0
        bt = 0
        for i in range(left, current_bin_timestamp + 1):
            ft = max(ft, llm_worker.time_bins_rt[i][0])
            bt = max(bt, llm_worker.time_bins_rt[i][1])
        return struct.pack("iff", llm_worker.queue_record[0], ft, bt)
    
    def read_data(client_socket):
        data = client_socket.recv(32)
        return struct.unpack("iff", data)

class AIMDScheduler(Scheduler):
    data_format = AIMDSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self):
        self.batch_size = 0
        self.key = True

    def schedule(self, intensity_data):
        # intensity of queries waiting for first token
        current_intensity, max_first_token_time, max_between_token_time = intensity_data

        print(max_first_token_time, max_between_token_time)
        # profiled 99% latency
        if (max_first_token_time > 0.58 or max_between_token_time > 0.09):
            self.batch_size = self.batch_size // 2
        elif (self.batch_size < self.max_batch_size):
            if(self.key):
                self.batch_size += 1
                self.key = False
            else:
                self.key = True
        if(self.batch_size == 0):
            time.sleep(0.5)
        return BasicSchedulePlan(self.batch_size)
    
    def serve(self, model_worker, test_time, rec):
        start_time_pointer = time.perf_counter()
        self.max_batch_size = model_worker.supported_batchsize[-1]
        time_limit = start_time_pointer + test_time
        while(True):
            if(time.perf_counter() > time_limit):
                break
            current_intensity = get_llm_intensity_unix(AIMDScheduler.data_format)
            schedule_plan = self.schedule(current_intensity)
            schedule_plan.execute(model_worker, rec)
