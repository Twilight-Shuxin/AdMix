import time
import json
import torch
from enum import IntEnum
from pydantic import Field
from typing import List, Optional
from profiling.scheduling.constants import FeatureModel
from profiling.scheduling.monitor import get_llm_intensity_unix
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.batch_predict.MLP_model import MLPPredictModel
from plotting.trace_plot import TracePlotData, TraceAxisData, plot_trace_graph
from profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, conn_get_data

class ObservationSingleBatchData(FeatureModel):
    scheduled_tokens: int = 0
    cached_tokens: int = 0
    latency: float = 0

    def pred_feature_data(self):
        return self.vector()[:-1]

    def get_pred_feature_len(self):
        return len(self.pred_feature_data())

class ObservationData(FeatureModel):
    NQ: int = 0
    waiting_prompt_tkns: int = 0
    processing: Optional[ObservationSingleBatchData] = None
    batch_data: List[ObservationSingleBatchData] = Field(default_factory=list)

    @classmethod
    def parse_raw(cls, payload) -> "ObservationData":
        (nq, waiting_prompt_tkns), processing, batch_data = json.loads(payload)
        return cls(
            NQ = nq,
            waiting_prompt_tkns = waiting_prompt_tkns,
            processing = ObservationSingleBatchData(**dict(zip(ObservationSingleBatchData.model_fields, processing))) if processing else None,
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

class BatchBaseScheduler(Scheduler):
    data_format = BatchSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    class Action(IntEnum):
        UP = 3
        FLAT = 2
        DOWN = 1

    def __init__(self):
        self.batch_size = 1
        self.max_batch_size = 32
        self.last_execution_time = 0
        self.time_interval = 0.1 # 100ms
        self.prediction_logs = []
        self.execution_logs = []
        self.llm_token_budget = 2048
        self.solo_time_sum = 0

    def scheduler_init(self):
        device = "cuda"
        input_dim = ObservationSingleBatchData().get_pred_feature_len()
        self.predictor = MLPPredictModel(input_dim, device, self.model_client.llm_model)
        self.max_process_speed = self.get_prediction(
            ObservationData(
                batch_data = [
                    ObservationSingleBatchData(
                        scheduled_tokens = self.llm_token_budget,
                        cached_tokens = 0
                    )
                ]
            )
        )[0] / self.llm_token_budget # process speed in milliseconds

        profiled_data = self.model_client.read_profiled_data()
        self.max_batch_size = self.model_client.supported_batchsize[-1]
        for i in reversed(self.model_client.supported_batchsize):
            if(profiled_data[str(i)]['latency'] < self.time_interval):
                self.max_batch_size = i
                break
        self.solo_start_batch_size = 1
        for i in reversed([bs for bs in self.model_client.supported_batchsize if bs <= self.max_batch_size]):
            if(profiled_data[str(i)]['sm_active'] < 50):
                self.solo_start_batch_size = i
                break
        self.alpha = self.model_client.alpha
        self.beta = self.model_client.beta

    def step_up(self):
        if(self.batch_size == 0):
            self.batch_size += 1
            return
        if(self.batch_size < self.max_batch_size):
            self.batch_size += self.alpha if (self.batch_size != 1 or self.alpha == 1) else self.alpha - 1
            self.batch_size = min(self.max_batch_size, self.batch_size)

    def step_down(self):
        self.batch_size = self.batch_size // self.beta

    def execute_action(self, action):
        match action:
            case self.Action.UP:
                self.step_up()
            case self.Action.FLAT:
                if(self.batch_size == 0):
                    self.step_up()
            case self.Action.DOWN:
                self.step_down()

    def get_prediction(self, observation_data):
        vals = observation_data.pred_feature_data()
        if(not vals):
            return []
        pred_latency = self.predictor.exec(vals)
        torch.cuda.synchronize()
        return pred_latency

    def _serve(self):
        self.scheduler_init()
        # May have remaining data in intensity server; fetch once to clear buffer
        current_intensity = get_llm_intensity_unix(self.data_format)
        self.start_serve()

        while(time.perf_counter() < self.model_client.runtime_data.time_limit):
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.schedule(current_intensity)

        self.model_client.write_csv("batch_predicted.csv",
            ["Scheduled Tkns", "Cached Tkns", "Real Latency", "Pred Latency"],
            self.prediction_logs)
        self.model_client.write_csv("batch_execution.csv",
            ["Time", "Batch size", "Details", "Processing", "Additional_data"],
            self.execution_logs
        )
        batch_size_data = TraceAxisData(TracePlotData('Batch Size', [(x[0], x[1]) for x in self.execution_logs]))
        plot_trace_graph(self.model_client.get_file_path('batch_size.png'), batch_size_data)