from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, pack_data_bytes, unpack_data_bytes, get_data_length
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from pydantic import BaseModel
import time

class ObservationSingleBatchData(FeatureModel):
    latency: float = 0
    ND: int = 0
    LD_avg: int = 0
    NP: int = 0
    LP_sum: int = 0

    def pred_feature_data(self):
        return self.vector()[1:]
    
    def get_pred_feature_len(self):
        return self.pred_feature_data()

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
        return unpack_data_bytes(ObservationData, client_socket, length)

class BatchScheduler(AIMDBaseScheduler):
    data_format = BatchSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self):
        self.batch_size = 1
        self.max_batch_size = 32
        self.last_execution_time = 0
        self.time_interval = 0.1 # 100ms
        
        input_dim = ObservationSingleBatchData().get_pred_feature_len()
        device = "cuda"
        self.predictor = None

    def get_prediction(self, observation_data):
        vals = observation_data.pred_feature_data()
        pred_latency = self.predictor.exec(vals)
        torch.cuda.synchronize()
        return pred_latency
    
    def get_action(self, real_latency, pred_latency):
        return self.Action.FLAT

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()
        pred_latency = self.get_prediction(intensity_data)
        action = self.get_action([x.latency for x in intensity_data.data], pred_latency)
        self.execute_action(action)
        self.execute()
    
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
    
    def _serve(self):
        self.model_worker.double_barrier.wait()
        self.model_worker.runtime_data.start_time_pointer = time.perf_counter()
        self.model_worker.runtime_data.time_limit = self.model_worker.runtime_data.start_time_pointer + self.model_worker.test_config.test_time
        
        while(time.perf_counter() < self.model_worker.runtime_data.time_limit):
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)

        
