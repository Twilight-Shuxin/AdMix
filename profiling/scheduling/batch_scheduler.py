import time
import json
import torch
from typing import List
from pydantic import BaseModel, Field
from profiling.scheduling.constants import FeatureModel
from profiling.models.workers.constants import SyncType
from profiling.scheduling.monitor import get_llm_intensity_unix
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.batch_predict.MLP_model import MLPPredictModel
from profiling.scheduling.scheduler import AIMDBaseScheduler, SchedulingDataFormat, pack_data_bytes, conn_get_data

class ObservationSingleBatchData(FeatureModel):
    cached_tokens: int = 0
    LD_avg: int = 0
    NP: int = 0
    scheduled_tokens: int = 0
    latency: float = 0

    def pred_feature_data(self):
        return self.vector()[:-1]
    
    def get_pred_feature_len(self):
        return len(self.pred_feature_data())

class ObservationData(FeatureModel):
    ND: int = 0
    NP: int = 0
    batch_data: List[ObservationSingleBatchData] = Field(default_factory=list)

    @classmethod
    def parse_raw(cls, payload) -> "ObservationData":
        general_data, batch_data = json.loads(payload)
        return cls(
            ND = general_data[0],
            NP = general_data[1],
            batch_data=[
                ObservationSingleBatchData(**dict(zip(ObservationSingleBatchData.model_fields, row)))
                for row in batch_data
            ]
        )
    
    def pred_feature_data(self):
        return [x.pred_feature_data() for x in self.batch_data]

class BatchSchedulingDataFormat(SchedulingDataFormat):
    @classmethod
    def default_data(cls):
        return ObservationData()

    @classmethod
    def read_data(cls, client_socket):
        return conn_get_data(ObservationData, client_socket, chunked = True)

class BatchScheduler(AIMDBaseScheduler):
    data_format = BatchSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self):
        self.batch_size = 1
        self.max_batch_size = 32
        self.last_execution_time = 0
        self.time_interval = 0.1 # 100ms
        
        device = "cuda"
        input_dim = ObservationSingleBatchData().get_pred_feature_len()
        self.predictor = MLPPredictModel(input_dim, device)
        self.prediction_logs = []
        self.execution_logs = []

    def get_prediction(self, observation_data):
        vals = observation_data.pred_feature_data()
        if(not vals):
            return []
        pred_latency = self.predictor.exec(vals)
        torch.cuda.synchronize()
        return pred_latency
    
    def get_action(self, intensity_data, pred_latency):
        # No batch information has been collected in the last scheduling interval
        if(not intensity_data.batch_data):
            # If no queries are under execution, increase batch size. 
            if(intensity_data.ND == 0 and intensity_data.NP == 0):
                return self.Action.UP
            # Else reduce batch size
            # TODO: Improve proactive method to reduce batch size
            return self.Action.DOWN

        real_latency = [x.latency for x in intensity_data.batch_data]
        percentages = [(r - p) / p for p, r in zip(pred_latency, real_latency) if r > p and p != 0]

        self.prediction_logs.extend([
                feature.copy() + [latency, pred_latency]
                for feature, latency, pred_latency in zip(
                    intensity_data.pred_feature_data(), real_latency,pred_latency,
                )
            ])
        
        if(not percentages):
            return self.Action.UP

        max_pct = max(percentages)
        avg_pct = sum(percentages) / len(percentages)

        if max_pct > 0.5 or avg_pct > 0.2:
            return self.Action.DOWN
        elif avg_pct < 0.15:
            return self.Action.UP
        return self.Action.FLAT

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()

        pred_latency = self.get_prediction(intensity_data)
        pred_latency = [float(x) for x in pred_latency]
        action = self.get_action(intensity_data, pred_latency)
        self.execute_action(action)
        
        current_timestamp = round(time.perf_counter() - self.model_client.runtime_data.start_time_pointer, 3)
        self.execution_logs.append([current_timestamp, self.batch_size, list(zip([x.latency for x in intensity_data.batch_data], pred_latency))])
        self.execute()
    
    def execute(self):
        end_round_time = self.start_round_time + self.time_interval
        schedule_plan = BasicSchedulePlan(self.batch_size)
        self.last_execution_time = 0
        while(time.perf_counter() + self.last_execution_time < end_round_time):
            start_time = time.perf_counter()
            self.model_client.broadcast_workers(schedule_plan, sync = SyncType.ALL)
            self.last_execution_time = time.perf_counter() - start_time
        left_time = end_round_time - time.perf_counter()
        if(left_time > 0.001):
            time.sleep(left_time)
    
    def _serve(self):
        self.scheduler_init()
        # May have remaining data in intensity server; fetch once to clear buffer
        current_intensity = get_llm_intensity_unix(self.data_format)
        self.start_serve()
        
        while(time.perf_counter() < self.model_client.runtime_data.time_limit):
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)

        self.model_client.write_csv("batch_predicted.csv", 
            ["Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn", "Latency", "Pred Latency"],
            self.prediction_logs)
        self.model_client.write_csv("batch_execution.csv", 
            ["Time", "Batch size", "Details"],
            self.execution_logs
        )
        
