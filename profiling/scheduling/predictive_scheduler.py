import torch
from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, conn_get_data
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.predict.MLP_model import MLPPredictModel
from profiling.scheduling.monitor import get_llm_intensity_unix
from pydantic import BaseModel, Field
from profiling.scheduling.constants import SCHEDULING_BIN_SIZE, FeatureModel
from plotting.constants import *
from profiling.constants import *
from enum import IntEnum
import time
import asyncio

def get_point_data(llm_worker):
    timestamp = time.perf_counter() - llm_worker.runtime_data.start_time_pointer
    NP = llm_worker.runtime_data.waiting_first_record_counter.counter
    LP_avg = 0
    LP_max = 0
    TP_maxL = 0
    if(NP):
        LP_avg = llm_worker.scheduler_data.total_prefill_length / NP
        LP_max, submitted_timestamp = llm_worker.scheduler_data.prefill_length_heap.top()
        TP_maxL = timestamp - submitted_timestamp
    ND = llm_worker.runtime_data.submitted_record_counter.counter - llm_worker.runtime_data.waiting_first_record_counter.counter
    LD_avg = 0
    if(ND):
        LD_avg = llm_worker.scheduler_data.total_decode_length / ND
    return ObservationPointData(
            prefill_data = PrefillRequestData(
                    N = NP, 
                    L_avg = LP_avg, 
                    L_max = LP_max, 
                    T_maxL = TP_maxL), 
            ND = ND, LD_avg = LD_avg)

def get_period_data(llm_worker):
    timestamp = time.perf_counter() - llm_worker.runtime_data.start_time_pointer
    
    NP = len(llm_worker.scheduler_data.prefill_length_heap_period)
    LP_avg = 0
    LP_max = 0
    TP_maxL = 0
    if(NP):
        LP_max, submitted_timestamp = llm_worker.scheduler_data.prefill_length_heap_period.top()
        LP_avg = llm_worker.scheduler_data.prefill_length_heap_period.get_total() / NP
        TP_maxL = timestamp - submitted_timestamp
    
    NF = llm_worker.scheduler_data.finished_count_period
    if(NF):
        LF_avg = llm_worker.scheduler_data.finished_tokens_period / NF
    else:
        LF_avg = 0
    
    TpP = llm_worker.scheduler_data.prefill_throughput_period
    TpD = llm_worker.scheduler_data.decode_throughput_period

    llm_worker.scheduler_data.prefill_throughput_period = 0
    llm_worker.scheduler_data.decode_throughput_period = 0 

    llm_worker.scheduler_data.prefill_length_period = 0
    llm_worker.scheduler_data.prefill_length_heap_period = MaxHeap()

    llm_worker.scheduler_data.finished_count_period = 0
    llm_worker.scheduler_data.finished_tokens_period = 0
    
    llm_worker.scheduler_data.refreshed_counter += 1
    if(llm_worker.scheduler_data.refreshed_counter > 10000):
        llm_worker.scheduler_data.refreshed_counter = 0

    return ObservationPeriodData(
                prefill_data = PrefillRequestData(
                    N=NP, 
                    L_avg=LP_avg, 
                    L_max=LP_max, 
                    T_maxL=TP_maxL), 
                NF = NF, LF_avg = LF_avg, 
                TpP = TpP, TpD = TpD)

def collect_data(llm_worker):
    point_data = get_point_data(llm_worker)
    period_data = get_period_data(llm_worker)
    return ObservationData(point_data = point_data, period_data = period_data)

async def predictive_profiler(llm_worker):
    logs = []
    last_point_data = ObservationPointData()
    try:
        print("Started profiling")
        while(True):
            last_timestamp = time.perf_counter() - llm_worker.runtime_data.start_time_pointer
            await asyncio.sleep(0.1)
            observation_data = collect_data(llm_worker)
            if(time.perf_counter() > llm_worker.runtime_data.time_limit):
                raise asyncio.CancelledError
            logs.append([round(last_timestamp, 4)] + last_point_data.vector() + observation_data.period_data.vector())
            last_point_data = observation_data.point_data
            
    except asyncio.CancelledError:
        llm_worker.write_csv("predict_data.csv", 
        PredHeaders.ORDERED, logs)
    except Exception as e:
        print(e)

class PrefillRequestData(FeatureModel):
        N: int = 0
        L_avg: float = 0
        L_max: int = 0
        T_maxL: float = 0
    
class ObservationPointData(FeatureModel):
    prefill_data: PrefillRequestData = Field(
        default_factory=PrefillRequestData
    )
    ND: int = 0
    LD_avg: float = 0

class ObservationPeriodData(FeatureModel):
    prefill_data: PrefillRequestData = Field(
        default_factory=PrefillRequestData
    )
    NF: int = 0
    LF_avg: float = 0
    TpP: int = 0
    TpD: int = 0

class ObservationData(FeatureModel):
    point_data: ObservationPointData = Field(
        default_factory=ObservationPointData
    )
    period_data: ObservationPeriodData = Field(
        default_factory=ObservationPeriodData
    )
    
    def pred_feature_data(self):
        return self.vector()[:-2]
    
    def get_pred_feature_len(self):
        return len(self.pred_feature_data())

class PredHeaders:
    time = "Time"
    prefill_N_t1 = "Prefill_N t1"
    prefill_avgL_t1 = "Prefill_avgL t1"
    prefill_maxL_t1 = "Prefill_maxL t1"
    prefill_T_maxL_t1 = "Prefill_T_maxL t1"
    decode_N_t1 = "Decode_N t1"
    decode_avgL_t1 = "Decode_avgL t1"
    prefill_N_t1_2 = "Prefill_N t1-2"
    prefill_avgL_t1_2 = "Prefill_avgL t1-2"
    prefill_maxL_t1_2 = "Prefill_maxL t1-2"
    prefill_T_maxL_t1_2 = "Prefill_T_maxL t1-2"
    finished = "Finished_N"
    finished_tkn = "Avg Finished_Tkn"
    prefill_tp = "Prefill_Tp"
    decode_tp = "Decode_Tp"

    ORDERED = [
        time,
        prefill_N_t1,
        prefill_avgL_t1,
        prefill_maxL_t1,
        prefill_T_maxL_t1,
        decode_N_t1,
        decode_avgL_t1,
        prefill_N_t1_2,
        prefill_avgL_t1_2,
        prefill_maxL_t1_2,
        prefill_T_maxL_t1_2,
        finished,
        finished_tkn,
        prefill_tp,
        decode_tp,
    ]

class PredictiveSchedulingDataFormat(SchedulingDataFormat):
    @classmethod
    def default_data(cls):
        return ObservationData()
    
    @classmethod
    def pack_data(cls, llm_worker):
        data = collect_data(llm_worker)
        return pack_data_bytes(data)

    @classmethod
    def read_data(cls, client_socket):
        return conn_get_data(ObservationData, client_socket)

class PredictiveScheduler(Scheduler):
    data_format = PredictiveSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan
    profiling_executor = predictive_profiler
    
    class Threshold(BaseModel):
        fine_threshold : float
        alert_threshold : float
        accumulate_threshold : int
        accumulated : int = 0
    
    class Action(IntEnum):
        UP = 3
        FLAT = 2
        DOWN = 1

    def __init__(self):
        self.batch_size = 1
        self.max_batch_size = 32
        self.last_execution_time = 0
        self.time_interval = 0.1 # 100ms
        
        input_dim = ObservationData().get_pred_feature_len()
        device = "cuda"
        self.prefill_predictor = MLPPredictModel("prefill", input_dim, device)
        self.decode_predictor = MLPPredictModel("decode", input_dim, device)
        
        self.threshold_D = self.Threshold(
            fine_threshold = 0.2,
            alert_threshold = 0.3,
            accumulate_threshold = 5
        )
        self.threshold_P = self.Threshold(
            fine_threshold = 0.2,
            alert_threshold = 0.3,
            accumulate_threshold = 5,
        )
        self.prediction_logs = []
        self.last_point_data = ObservationPointData()

    def get_prediction(self, observation_data):
        vals = observation_data.pred_feature_data()
        pred_TpP = self.prefill_predictor.exec([vals])
        pred_TpD = self.decode_predictor.exec([vals])
        torch.cuda.synchronize()
        pred_TpP = pred_TpP.item()
        pred_TpD = pred_TpD.item()
        return max(0, round(pred_TpP)), max(0, round(pred_TpD))

    def check_threshold(self, threshold, pred_Tp, true_Tp):
        if(true_Tp >= pred_Tp - max(2, pred_Tp * threshold.fine_threshold)):
            if(true_Tp == 0 and pred_Tp != 0):
                threshold.accumulated += 1
                if(pred_Tp > 2):
                    return self.Action.FLAT
                else:
                    return self.Action.UP 
            else:
                threshold.accumulated = 0
            return self.Action.UP
        if(true_Tp < pred_Tp - max(3, pred_Tp * threshold.alert_threshold)):
            threshold.accumulated = 0
            return self.Action.DOWN
        threshold.accumulated += 1
        if(threshold.accumulated > threshold.accumulate_threshold):
            threshold.accumulated = 0
            return self.Action.DOWN
        return self.Action.FLAT

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()

        # intensity of queries waiting for first token
        observation_data = ObservationData(point_data = self.last_point_data, period_data = intensity_data.period_data)
        self.last_point_data = intensity_data.point_data
        pred_TpP, pred_TpD = self.get_prediction(observation_data)
        action = self.check_threshold(self.threshold_P, pred_TpP, intensity_data.period_data.TpP)
        action = min(action, self.check_threshold(self.threshold_D, pred_TpD, intensity_data.period_data.TpD))
        match action:
            case self.Action.UP:
                self.step_up()
            case self.Action.FLAT:
                if(self.batch_size == 0):
                    self.step_up()
            case self.Action.DOWN:  
                self.step_down()
        self.prediction_logs.append([round(time.perf_counter() - self.model_worker.runtime_data.start_time_pointer, 3),
            self.batch_size, intensity_data.period_data.TpP, pred_TpP, intensity_data.period_data.TpD, pred_TpD])
        self.execute()
    
    def step_up(self):
        if(self.batch_size == 0):
            self.batch_size += 1
            return
        if(self.batch_size < self.max_batch_size):
            self.batch_size += self.alpha if (self.batch_size != 1 or self.alpha == 1) else self.alpha - 1
            self.batch_size = min(self.max_batch_size, self.batch_size)
    
    def step_down(self):
        self.batch_size = self.batch_size // self.beta
    
    def execute(self):
        end_round_time = self.start_round_time + self.time_interval
        schedule_plan = BasicSchedulePlan(self.batch_size)
        self.last_execution_time = 0
        while(time.perf_counter() + self.last_execution_time < end_round_time):
            start_time = time.perf_counter()
            schedule_plan.execute(self.model_worker)
            self.last_execution_time = time.perf_counter() - start_time
        left_time = end_round_time - time.perf_counter()
        if(left_time > 0.001):
            time.sleep(left_time)
    
    def scheduler_init(self):
        profiled_data = self.model_worker.read_profiled_data()
        max_batch_size = self.model_worker.supported_batchsize[-1]
        for i in reversed(range(1, max_batch_size + 1)):
            if(profiled_data[str(i)] < self.time_interval):
                max_batch_size = i
                break
        self.max_batch_size = max_batch_size
        self.alpha = self.model_worker.alpha
        self.beta = self.model_worker.beta
    
    def _serve(self):
        self.scheduler_init()

        self.model_worker.double_barrier.wait()
        self.model_worker.runtime_data.start_time_pointer = time.perf_counter()
        self.model_worker.runtime_data.time_limit = self.model_worker.runtime_data.start_time_pointer + self.model_worker.test_config.test_time

        while(time.perf_counter() < self.model_worker.runtime_data.time_limit):
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)
        
        self.model_worker.write_csv("scheduled.csv", 
            ["Time", "Batch Size", "Prefill_Tp", "Predicted_Prefill_Tp", "Decode_Tp", "Predicted_Decode_Tp"],
            self.prediction_logs)
        

        