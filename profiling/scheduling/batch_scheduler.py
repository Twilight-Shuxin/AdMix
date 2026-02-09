import time
from typing import List
from pydantic import BaseModel, Field
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.scheduler import AIMDBaseScheduler, SchedulingDataFormat, pack_data_bytes, conn_get_data
from profiling.scheduling.batch_predict.MLP_model import MLPPredictModel
from profiling.scheduling.constants import FeatureModel
from profiling.scheduling.monitor import get_llm_intensity_unix

class ObservationSingleBatchData(FeatureModel):
    latency: float = 0
    ND: int = 0
    LD_avg: int = 0
    NP: int = 0
    LP_sum: int = 0

    def pred_feature_data(self):
        return self.vector()[1:]
    
    def get_pred_feature_len(self):
        return len(self.pred_feature_data())

class ObservationData(FeatureModel):
    data: List[ObservationSingleBatchData] = Field(default_factory=list)

    @classmethod
    def parse_raw(cls, raw: list[list[ObservationSingleBatchData]]) -> "ObservationData":
        return cls(
            data=ObservationSingleBatchData(
                **dict(zip(ObservationSingleBatchData.model_fields.keys(), values))
            )
        )
    
    def pred_feature_data(self):
        return [x.pred_feature_data() for x in self.data]

class BatchSchedulingDataFormat(SchedulingDataFormat):
    @classmethod
    def default_data(cls):
        return ObservationData()

    @classmethod
    def read_data(cls, client_socket):
        length = get_data_length(client_socket)
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
        pred_latency = self.predictor.exec(vals)
        torch.cuda.synchronize()
        return pred_latency
    
    def get_action(self, real_latency, pred_latency):
        return self.Action.FLAT

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()
        if(intensity_data.data):
            print("??", intensity_data.data)
            pred_latency = self.get_prediction(intensity_data)
            self.prediction_logs.extend(
                [
                    feature + [latency, pred_latency]
                    for feature, latency, pred_latency in zip(
                        observation_data.pred_feature_data(),
                        [x.latency for x in intensity_data.data],
                        pred_latency,
                    )
                ]
            )

            action = self.get_action([x.latency for x in intensity_data.data], pred_latency)
        else:
            action = self.Action.UP
        self.execute_action(action)
        self.execute()
    
    def execute(self):
        end_round_time = self.start_round_time + self.time_interval
        schedule_plan = BasicSchedulePlan(self.batch_size)
        # self.last_execution_time = 0
        # while(time.perf_counter() + self.last_execution_time < end_round_time):
        #     start_time = time.perf_counter()
        #     schedule_plan.execute(self.model_worker)
        #     self.last_execution_time = time.perf_counter() - start_time
        left_time = end_round_time - time.perf_counter()
        if(left_time > 0.001):
            time.sleep(left_time)
    
    def _serve(self):
        self.scheduler_init()
        self.model_worker.double_barrier.wait()
        self.model_worker.runtime_data.start_time_pointer = time.perf_counter()
        self.model_worker.runtime_data.time_limit = self.model_worker.runtime_data.start_time_pointer + self.model_worker.test_config.test_time
        
        while(time.perf_counter() < self.model_worker.runtime_data.time_limit):
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)

        self.model_worker.write_list_to_csv("batch_predicted.csv", 
            ["Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn", "Latency", "Pred Latency"],
            self.prediction_logs)

        
