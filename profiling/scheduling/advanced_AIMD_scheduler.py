from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.monitor import get_llm_intensity_unix
from profiling.scheduling.constants import *
from plotting.constants import *
from profiling.constants import *
from enum import Enum
from profiling.constants import MaxHeapWithTimestamp, AvgDequeWithTimestamp
import struct
import time

class AdvancedAIMDSchedulingDataFormat(SchedulingDataFormat):
    def default_data():
        return 0,0,0,0,0.0
    
    def pack_data(llm_worker):
        if(llm_worker.btt_cnt_rt == 0):
            avg_btt = 0
        else:
            avg_btt = llm_worker.total_btt_rt / llm_worker.btt_cnt_rt
        data = struct.pack("iiiif", llm_worker.queue_record[0], llm_worker.queue_record[1], llm_worker.btt_cnt_rt, llm_worker.ftt_cnt_rt, avg_btt)
        llm_worker.total_btt_rt = 0
        llm_worker.btt_cnt_rt = 0
        llm_worker.ftt_cnt_rt = 0
        return data

    def read_data(client_socket):
        data = client_socket.recv(32)
        return struct.unpack("iiiif", data)

class AdvancedAIMDScheduler(Scheduler):
    data_format = AdvancedAIMDSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    class SchedulerState(Enum):
        MONITORING = 1
        SCHEDULING = 2
        SOLO = 3

    def __init__(self):
        self.timestamp = 0
        self.batch_size = 1
        self.max_batch_size = 32
        self.last_batch_size = 0
        self.last_execution_time = 0
        self.max_exec_cnt = 1
        self.window_btt = AvgDequeWithTimestamp()
        self.window_prefill_heap = MaxHeapWithTimestamp()
        self.window_decode_heap = MaxHeapWithTimestamp()
        self.window_btt_heap = MaxHeapWithTimestamp()
        self.max_window_btt_size = 15
        self.stable_window_btt_size = 10
        self.state = self.SchedulerState.MONITORING
        self.last_state = None
        self.time_interval = 0.1 # 100ms
        self.last_intensity = None
        self.aggregated_latency = 0
        self.modified_flag = False
    
    def step_in_solo(self, intensity_data):
        self.last_state = self.state
        self.state = self.SchedulerState.SOLO
        self.last_intensity_data = intensity_data
        self.last_batch_size = self.batch_size
        self.batch_size = self.max_batch_size

    def step_out_solo(self, intensity_data):
        self.state = self.last_state
        self.batch_size = self.last_batch_size

    def step_up(self):
        if((self.max_exec_cnt + 1) * self.last_execution_time > self.time_interval):
            if(self.batch_size < self.max_batch_size):
                self.batch_size += 1
                self.max_exec_cnt //= 2
        else:
            self.max_exec_cnt += 1
    
    def step_down(self):
        if (self.max_exec_cnt // 2 > 0):
            self.max_exec_cnt = self.max_exec_cnt // 2
        else:
            self.batch_size = self.batch_size // 2
        if(self.batch_size == 0 or self.max_exec_cnt == 0):
            self.set_monitoring_status()
    
    def set_monitoring_status(self):
        self.state = self.SchedulerState.MONITORING
        self.batch_size = 1
        self.max_exec_cnt = 1
        self.window_btt_heap.clear()
        self.window_btt.clear()
        self.window_decode_heap.clear()
        self.window_prefill_heap.clear()
    
    # simply perform the operation of adding to window_btt
    def add_to_window(self, intensity_data, timestamp):
        decode_cnt, prefill_cnt, btt_cnt, ftt_cnt, avg_btt = intensity_data
        if(self.modified_flag):
            return
        if(avg_btt < EPS):
            return
        self.window_btt.append((avg_btt, timestamp))
        self.window_btt_heap.push((avg_btt, timestamp))
        self.window_decode_heap.push((decode_cnt, timestamp))
        self.window_prefill_heap.push((prefill_cnt, timestamp))

    def evict_from_window(self):
        _, oldest_timestamp = self.window_btt.evict()
        self.window_btt_heap.set_mintimestamp(oldest_timestamp)
        self.window_decode_heap.set_mintimestamp(oldest_timestamp)
        self.window_prefill_heap.set_mintimestamp(oldest_timestamp)

    def schedule(self, intensity_data):
        # intensity of queries waiting for first token
        decode_cnt, prefill_cnt, btt_cnt, ftt_cnt, avg_btt = intensity_data

        if(decode_cnt != 0 and avg_btt < EPS):
            self.aggregated_latency += self.time_interval
        else:
            self.aggregated_latency = 0


        if(prefill_cnt == 0 and decode_cnt == 0 and self.state != self.SchedulerState.SOLO):
            self.step_in_solo(intensity_data)
            # formatted_values = tuple(
            #     float(f"{v:.2f}") if isinstance(v, float) else v for v in intensity_data
            # )

            # print(f"{time.perf_counter() - self.start_timepointer:.2f}", 
            #     self.state, 
            #     len(self.window_btt), 
            #     formatted_values)
        
        if(self.state == self.SchedulerState.SOLO):
            # print(intensity_data, "hi")
            if(prefill_cnt != 0 or decode_cnt != 0):
                self.step_out_solo(intensity_data)
                # print("out")
            else:
                self.execute_single()
                return
        
        # formatted_values = tuple(
        #     float(f"{v:.2f}") if isinstance(v, float) else v for v in intensity_data
        # )

        # print(f"{time.perf_counter() - self.start_timepointer:.2f}", 
        #     self.state, (self.batch_size, self.max_exec_cnt),
        #     len(self.window_btt), 
        #     formatted_values)
        
        # time.sleep(self.time_interval)
        # return
        self.modified_flag = False
        if(self.aggregated_latency > EPS):
            self.modified_flag = True
            avg_btt = self.aggregated_latency
        if(self.state == self.SchedulerState.SCHEDULING and avg_btt > self.window_btt.avg() and len(self.window_btt) >= self.max_window_btt_size):
                self.set_monitoring_status()

        if(self.state == self.SchedulerState.MONITORING and len(self.window_btt) < self.stable_window_btt_size):
            self.add_to_window(intensity_data, self.timestamp)
            time.sleep(self.time_interval)
        else:
            self.state = self.SchedulerState.SCHEDULING
            # print("-->", avg_btt, self.window_btt.avg(), self.window_btt_heap.peek()[0], self.window_btt_heap.peek()[0] * 1.2)
            if(avg_btt > self.window_btt_heap.peek()[0]):
                self.step_down()  
            elif(avg_btt > self.window_btt.avg() or self.modified_flag):
                self.add_to_window(intensity_data, self.timestamp)
            else: 
                self.step_up()
                self.add_to_window(intensity_data, self.timestamp)
                self.evict_from_window()
            self.execute()
        self.timestamp += self.time_interval
    
    def execute_single(self):
        schedule_plan = BasicSchedulePlan(self.batch_size)
        schedule_plan.execute(self.model_worker, self.rec)
    
    def execute(self):
        start_round_time = time.perf_counter()
        end_round_time = start_round_time + self.time_interval
        cnt = 0
        schedule_plan = BasicSchedulePlan(self.batch_size)
        while(time.perf_counter() < end_round_time and cnt < self.max_exec_cnt):
            start_time = time.perf_counter()
            schedule_plan.execute(self.model_worker, self.rec)
            self.last_execution_time = time.perf_counter() - start_time
            cnt += 1
        left_time = end_round_time - time.perf_counter()
        if(left_time > 0.01):
            time.sleep(left_time)
    
    def scheduler_init(self):
        profile_data = self.model_worker.read_profile_data()
        max_batch_size = self.model_worker.supported_batchsize[-1]
        for i in reversed(range(1, max_batch_size + 1)):
            if(profile_data[str(i)] < self.time_interval):
                max_batch_size = i
                break
        self.max_batch_size = max_batch_size
    
    def serve(self, model_worker, test_time, rec):
        time_limit = time.perf_counter() + test_time
        self.start_timepointer = time.perf_counter()
        self.model_worker = model_worker
        self.scheduler_init()
        self.rec = rec
        model_worker.single_barrier.wait()
        while(True):
            if(time.perf_counter() > time_limit):
                break
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)

        

        