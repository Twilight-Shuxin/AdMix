from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, unpack_data_bytes, get_data_length
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.predict.MLP_model import MLPPredictModel
from profiling.scheduling.monitor import get_llm_intensity_unix
from pydantic import BaseModel
from profiling.scheduling.constants import *
from plotting.constants import *
from profiling.constants import *
from enum import Enum, IntEnum
from profiling.constants import MaxHeapWithTimestamp, AvgDequeWithTimestamp
import struct
import time
import asyncio

# Data used for prediction: 
# ==============================
#   t1 prefill intensity
#   t1 average prefill max length
#   t1 prefill max length
#   t1 decoding intensity
#   t1 decode average length
#   before t1 max length query execution time
#   t1-t2 prefill intensity
#   t1-t2 average prefill max length
#   t1-t2 prefill max length
#   t1-t2 max length query execution time
#   t1~t2 finished
#   t1~t2 finished average throughput
# ==============================

def get_point_data(llm_worker):
    timestamp = time.perf_counter() - llm_worker.start_time_pointer
    t1_prefill_cnt = llm_worker.queue_record[1]
    t1_prefill_avg_length = 0
    t1_max_prefill_length = 0
    t1_prefill_max_exec_time = 0
    if(t1_prefill_cnt != 0):
        t1_prefill_avg_length = llm_worker.total_prefill_length / t1_prefill_cnt
        t1_max_prefill_length, t1_prefill_max_exec_timestamp = llm_worker.prefill_length_heap.top()
        t1_prefill_max_exec_time = timestamp - t1_prefill_max_exec_timestamp
    t1_decode_cnt = llm_worker.decode_intensity
    t1_decode_avg_length = 0
    if(t1_decode_cnt != 0):
        t1_decode_avg_length = llm_worker.total_decode_length / t1_decode_cnt
    return (t1_prefill_cnt, t1_prefill_avg_length, t1_max_prefill_length, t1_prefill_max_exec_time, t1_decode_cnt, t1_decode_avg_length)

def get_period_data(llm_worker):
    timestamp = time.perf_counter() - llm_worker.start_time_pointer
    prefill_cnt_tmp = llm_worker.prefill_intensity_tmp
    prefill_length_tmp = 0
    max_prefill_length_tmp = 0
    prefill_max_exec_time_tmp = 0
    if(prefill_cnt_tmp != 0):
        prefill_length_tmp = (llm_worker.total_prefill_length_tmp / prefill_cnt_tmp)
        max_prefill_length_tmp = -llm_worker.prefill_length_heap_tmp[0][0]
        prefill_max_exec_time_tmp = (timestamp - llm_worker.prefill_length_heap_tmp[0][1])
    finished_cnt = llm_worker.finished_tmp
    if(finished_cnt > 0):
        finished_throughput_avg = llm_worker.finished_token_tmp / finished_cnt
    else:
        finished_throughput_avg = 0
    llm_worker.finished_tmp = 0
    llm_worker.finished_token_tmp = 0
    llm_worker.refreshed_flag += 1
    llm_worker.prefill_length_heap_tmp = []
    if(llm_worker.refreshed_flag > 10000):
        llm_worker.refreshed_flag = 0
    throughput_prefill = llm_worker.ftt_cnt_rt
    throughput_decode = llm_worker.btt_cnt_rt
    llm_worker.prefill_intensity_tmp = 0
    llm_worker.total_prefill_length_tmp = 0
    llm_worker.prefill_start_time_tmp = 0
    llm_worker.ftt_cnt_rt = 0
    llm_worker.btt_cnt_rt = 0
    return (prefill_cnt_tmp, prefill_length_tmp, max_prefill_length_tmp, prefill_max_exec_time_tmp, finished_cnt, finished_throughput_avg, throughput_prefill, throughput_decode)

async def collect_data(llm_worker):
    (
        t1_prefill_cnt,
        t1_prefill_avg_length,
        t1_max_prefill_length,
        t1_prefill_max_exec_time,
        t1_decode_cnt,
        t1_decode_avg_length,
    ) = get_point_data(llm_worker)
    await asyncio.sleep(0.1)
    (
        prefill_cnt_tmp, 
        prefill_length_tmp, 
        max_prefill_length_tmp, 
        prefill_max_exec_time_tmp,
        finished_cnt, 
        finished_throughput_avg, 
        throughput_prefill, 
        throughput_decode,
    ) = get_period_data(llm_worker)
    return  PredictiveSchedulingDataFormat.ObservationData(
        NP1 = t1_prefill_cnt,
        LP_avg1 = t1_prefill_avg_length,
        LP_max1 = t1_max_prefill_length,
        TP_maxL1 = t1_prefill_max_exec_time,
        ND1 = t1_decode_cnt,
        LD_avg1 = t1_decode_avg_length,
        NP12 = prefill_cnt_tmp,
        LP_avg12 = prefill_length_tmp,
        LP_max12 = max_prefill_length_tmp,
        TP_maxL12 = prefill_max_exec_time_tmp,
        NF_12 = finished_cnt,
        TpF_avg12 = finished_throughput_avg,
        TpP = throughput_prefill,
        TpD = throughput_decode
    )

async def predictive_profiler(llm_worker, test_config):
    logs = []
    try:
        print("started profiling")
        while(True):
            last_timestamp = time.perf_counter() - llm_worker.start_time_pointer
            data = await collect_data(llm_worker)
            timestamp = time.perf_counter() - llm_worker.start_time_pointer
            if(timestamp > llm_worker.time_limit):
                raise asyncio.CancelledError
            logs.append([round(last_timestamp, 4), data.NP1, data.LP_avg1, data.LP_max1, data.TP_maxL1,
                data.ND1, data.LD_avg1,
                data.NP12, data.LP_avg12, data.LP_max12, data.TP_maxL12,
                data.NF_12, data.TpF_avg12,
                data.TpP, data.TpD
            ])
            
    except asyncio.CancelledError:
        llm_worker.write_list_to_csv(test_config, "predict_data.csv", 
        ['Timestamp (s)', 't1 prefill intensity', 't1 prefill average length', 't1 prefill max length', 't1 prefill max length execution time (s)',
        't1 decode intensity', 't1 decode average length', 
        't1~t2 prefill intensity', 't1~t2 prefill average length', 't1~t2 prefill max length', 't1~t2 prefill max length execution time (s)',
        't1~t2 finished', 't1~t2 finished average throughput',
        'Prefill throughput', 'Decode throughput'], logs)
    except Exception as e:
        print(e)


class PredictiveSchedulingDataFormat(SchedulingDataFormat):
    class ObservationData(BaseModel):
        NP1: int = 0
        LP_avg1: float = 0
        LP_max1: int = 0
        TP_maxL1: float = 0
        ND1: int = 0
        LD_avg1: float = 0
        NP12: int = 0
        LP_avg12: float = 0
        LP_max12: int = 0
        TP_maxL12: float = 0
        NF_12: int = 0
        TpF_avg12: float = 0
        TpP: int = 0
        TpD: int = 0
    
    class MonitoredData(BaseModel):
        NP2: int = 0
        LP_avg2: float = 0
        LP_max2: int = 0
        TP_maxL2: float = 0
        ND2: int = 0
        LD_avg2: float = 0
        NP12: int = 0
        LP_avg12: float = 0
        LP_max12: int = 0
        TP_maxL12: float = 0
        NF_12: int = 0
        TpF_avg12: float = 0
        TpP: int = 0
        TpD: int = 0

    def default_data():
        return PredictiveSchedulingDataFormat.MonitoredData()
    
    def pack_data(llm_worker):
        (
            t2_prefill_cnt,
            t2_prefill_avg_length,
            t2_max_prefill_length, 
            t2_prefill_max_exec_time,
            t2_decode_cnt,
            t2_decode_avg_length,
        ) = get_point_data(llm_worker)
        (
            prefill_cnt_tmp, 
            prefill_length_tmp, 
            max_prefill_length_tmp, 
            prefill_max_exec_time_tmp,
            finished_cnt,
            finished_throughput_avg, 
            throughput_prefill, 
            throughput_decode,
        ) = get_period_data(llm_worker)
        data = PredictiveSchedulingDataFormat.MonitoredData(
            NP2 = t2_prefill_cnt,
            LP_avg2 = t2_prefill_avg_length,
            LP_max2 = t2_max_prefill_length,
            TP_maxL2 = t2_prefill_max_exec_time,
            ND2 = t2_decode_cnt,
            LD_avg2 = t2_decode_avg_length,
            NP12 = prefill_cnt_tmp,
            LP_avg12 = prefill_length_tmp,
            LP_max12 = max_prefill_length_tmp,
            TP_maxL12 = prefill_max_exec_time_tmp,
            NF_12 = finished_cnt, 
            TpF_avg12 = finished_throughput_avg,
            TpP = throughput_prefill,
            TpD = throughput_decode,
        )
        return pack_data_bytes(data)

    def read_data(client_socket):
        length = get_data_length(client_socket)
        return unpack_data_bytes(PredictiveSchedulingDataFormat.MonitoredData, client_socket, length)

class PredictiveScheduler(Scheduler):
    data_format = PredictiveSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan
    profiling_executor = predictive_profiler

    class SchedulerState(Enum):
        SCHEDULING = 1
        SOLO = 2
    
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
        self.timestamp = 0
        self.batch_size = 1
        self.max_batch_size = 32
        self.last_batch_size = 0
        self.last_execution_time = 0
        self.max_exec_cnt = 1
        self.last_state = None
        self.time_interval = 0.1 # 100ms
        self.last_intensity = None
        self.aggregated_latency = 0
        self.modified_flag = False
        input_dim = 11
        device = "cuda"
        self.monitored_data = PredictiveSchedulingDataFormat.MonitoredData()
        self.prefill_predictor = MLPPredictModel("prefill", input_dim, device)
        self.decode_predictor = MLPPredictModel("decode", input_dim, device)
        self.threshold_D = PredictiveScheduler.Threshold(
            fine_threshold = 0.2,
            alert_threshold = 0.3,
            accumulate_threshold = 5
        )
        self.threshold_P = PredictiveScheduler.Threshold(
            fine_threshold = 0.2,
            alert_threshold = 0.3,
            accumulate_threshold = 5,
        )
        self.prediction_logs_prefill = []
        self.prediction_logs_decode = []
        self.prediction_logs = []

    def get_prediction(self, observation_data):
        vals = list(observation_data.dict().values())[:-2]
        vals.extend([
            observation_data.LP_avg1 ** 2,
            observation_data.LP_max1 ** 2,
            observation_data.LD_avg1 ** 2,
            observation_data.LP_avg12 ** 2,
            observation_data.LP_max12 ** 2,
        ])
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
                    return PredictiveScheduler.Action.FLAT
                else:
                    return PredictiveScheduler.Action.UP 
            else:
                threshold.accumulated = 0
            return PredictiveScheduler.Action.UP
        if(true_Tp < pred_Tp - max(3, pred_Tp * threshold.alert_threshold)):
            threshold.accumulated = 0
            return PredictiveScheduler.Action.DOWN
        threshold.accumulated += 1
        if(threshold.accumulated > threshold.accumulate_threshold):
            threshold.accumulated = 0
            return PredictiveScheduler.Action.DOWN
        return PredictiveScheduler.Action.FLAT

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()
        # intensity of queries waiting for first token
        observation_data = PredictiveSchedulingDataFormat.ObservationData(
            NP1 = self.monitored_data.NP2,
            LP_avg1 = self.monitored_data.LP_avg2,
            TP_avg1 = self.monitored_data.TP_avg2,
            ND1 = self.monitored_data.ND2,
            LD_avg1 = self.monitored_data.LD_avg2,
            NP12 = intensity_data.NP12,
            LP_avg12 = intensity_data.LP_avg12,
            TP_avg12 = intensity_data.TP_avg12,
            NF_12 = intensity_data.NF_12,
            TpF_avg12 = intensity_data.TpF_avg12,
            TpP = intensity_data.TpP,
            TpD = intensity_data.TpD,
        )
        self.monitored_data = intensity_data
        pred_TpP, pred_TpD = self.get_prediction(observation_data)
        action = self.check_threshold(self.threshold_P, pred_TpP, intensity_data.TpP)
        action = min(action, self.check_threshold(self.threshold_D, pred_TpD, intensity_data.TpD))
        # self.prediction_logs_prefill.append((intensity_data.TpP, pred_TpP))
        # self.prediction_logs_decode.append((intensity_data.TpD, pred_TpD))
        match action:
            case PredictiveScheduler.Action.UP:
                self.step_up()
            case PredictiveScheduler.Action.FLAT:
                if(self.batch_size == 0):
                    self.step_up()
            case PredictiveScheduler.Action.DOWN:  
                self.step_down()
        self.prediction_logs.append([round(time.perf_counter() - self.start_time_pointer, 3), self.batch_size, intensity_data.TpP, pred_TpP, intensity_data.TpD, pred_TpD])
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
        # if(False):
        schedule_plan = BasicSchedulePlan(self.batch_size)
        self.last_execution_time = 0
        while(time.perf_counter() + self.last_execution_time < end_round_time):
            start_time = time.perf_counter()
            schedule_plan.execute(self.model_worker, self.rec)
            self.last_execution_time = time.perf_counter() - start_time
        left_time = end_round_time - time.perf_counter()
        if(left_time > 0.001):
            time.sleep(left_time)
    
    def scheduler_init(self):
        profile_data = self.model_worker.read_profile_data()
        max_batch_size = self.model_worker.supported_batchsize[-1]
        for i in reversed(range(1, max_batch_size + 1)):
            if(profile_data[str(i)] < self.time_interval):
                max_batch_size = i
                break
        self.max_batch_size = max_batch_size
        self.alpha = self.model_worker.alpha
        self.beta = self.model_worker.beta

    def save_logs(self):
        output_dir = "/home/k43015/MPSTest/data/real_pred"
        os.makedirs(output_dir, exist_ok=True)

        # define file paths
        prefill_path = os.path.join(output_dir,"-".join(["prefill", self.model_worker.test_config.llm_tag]) + ".txt")
        decode_path = os.path.join(output_dir, "-".join(["decode", self.model_worker.test_config.llm_tag]) + ".txt")

        # write prefill data
        with open(prefill_path, "w") as f:
            for real, pred in self.prediction_logs_prefill:
                f.write(f"{real} {pred}\n")

        # write decode data
        with open(decode_path, "w") as f:
            for real, pred in self.prediction_logs_decode:
                f.write(f"{real} {pred}\n")
        print(f"Saved to:\n{prefill_path}\n{decode_path}")
    
    def serve(self, model_worker, test_time, rec):
        time_limit = time.perf_counter() + test_time
        self.model_worker = model_worker
        self.scheduler_init()
        self.rec = rec
        model_worker.double_barrier.wait()
        self.start_time_pointer = time.perf_counter()
        model_worker.start_time_pointer = time.perf_counter()
        while(True):
            if(time.perf_counter() > time_limit):
                break
            start_time = time.perf_counter()
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)
        # self.save_logs()
        self.model_worker.write_list_to_csv(self.model_worker.test_config, "scheduled.csv", 
            ["Time", "Batch Size", "Prefill_Tp", "Predicted_Prefill_Tp", "Decode_Tp", "Predicted_Decode_Tp"],
            self.prediction_logs)
        

        