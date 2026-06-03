import json
import random
import time
import torch
from tqdm import tqdm
from typing import List, Optional
from pydantic import Field
from admix.profiling.models.workers.constants import SyncType
from admix.profiling.scheduling.constants import FeatureModel
from admix.profiling.scheduling.monitor import get_llm_intensity_unix
from admix.profiling.scheduling.schedule_plan import ContextSchedulePlan
from admix.plotting.trace_plot import TracePlotData, TraceAxisData, plot_trace_graph
from admix.profiling.scheduling.context_predict.MLP_model import MLPPredictModel
from admix.profiling.scheduling.scheduler import Scheduler, SchedulingDataFormat, conn_get_data

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

class ContextSchedulingDataFormat(SchedulingDataFormat):
    @classmethod
    def default_data(cls):
        return ObservationData()

    @classmethod
    def read_data(cls, client_socket):
        return conn_get_data(ObservationData, client_socket, chunked = True)

def context_predict(predictor, observation_data):
    vals = observation_data.pred_feature_data()
    if not vals:
        return []
    result = predictor.exec(vals)
    torch.cuda.synchronize()
    return result

class ContextScheduler(Scheduler):
    data_format = ContextSchedulingDataFormat
    scheduling_plan = ContextSchedulePlan

    def __init__(self):
        self.prediction_logs = []
        self.execution_logs = []
        self.llm_token_budget = 2048
        self.solo_mode = False
        self.time_interval = 0.05 # 50ms
        self.current_plan = None
        self.normal_plan = None
        self.pre_solo_mps = None
        self.observed_percentages = []
        self.MPS_step              = 5
        self.context_switch_interval = 10  # seconds
        self.last_context_switch_time = None
        self.pause = False

    def scheduler_init(self):
        device = "cuda"
        input_dim = ObservationSingleBatchData().get_pred_feature_len()
        self.predictor = MLPPredictModel(input_dim, device, self.model_client.llm_model)
        self.max_process_speed = context_predict(
            self.predictor,
            ObservationData(
                batch_data = [
                    ObservationSingleBatchData(
                        scheduled_tokens = self.llm_token_budget,
                        cached_tokens = 0
                    )
                ]
            )
        )[0] / self.llm_token_budget # process speed in milliseconds
        self.model_client.command_workers("context_setup", sync = SyncType.ALL)
        profiled_data = self.model_client.read_profiled_data()
        self.max_batch_size = self.model_client.supported_batchsize[-1]
        for i in reversed(self.model_client.supported_batchsize):
            if(profiled_data[str(i)]['latency'] < 0.05):
                self.max_batch_size = i
                break
        for i in reversed(self.model_client.supported_batchsize):
            if(profiled_data[str(i)]['latency'] < 0.1):
                self.max_batch_size_full = i
                break
        self.pre_solo_mps = 20
        self.prefill_estimate_time = 0
        self.normal_plan = ContextSchedulePlan(self.pre_solo_mps, self.max_batch_size)
        self.last_context_switch_time = time.perf_counter()

    def get_scheduling_plan(self, intensity_data, pred_latency):
        current_mps = self.current_plan.mps if self.current_plan else self.normal_plan.mps

        self.prefill_estimate_time = (int(intensity_data.waiting_prompt_tkns * self.max_process_speed), 0)
        processed_latency, processing_pred_latency = 0, 0
        if(intensity_data.processing):
            processed_latency, processing_pred_latency = intensity_data.batch_data.pop().latency, pred_latency.pop()
        real_latency = [x.latency for x in intensity_data.batch_data]
        if(self.pause == False and not self.solo_mode):
            self.observed_percentages.extend([(r - p) / p if r - 3 > p and p > 0 else 0 for r, p in zip(real_latency, pred_latency)])
            if len(self.observed_percentages) > 20:
                self.observed_percentages = self.observed_percentages[-20:]

        # Log predictions
        if intensity_data.batch_data:
            self.prediction_logs.extend([
                feature.copy() + [latency, pred_lat]
                for feature, latency, pred_lat in zip(
                    intensity_data.pred_feature_data(), real_latency, pred_latency,
                )
            ])

        if intensity_data.NQ == 0: # No running queries — run solo
            if not self.solo_mode:
                self.pre_solo_mps = current_mps
                self.solo_mode = True
            self.pause = False
            return ContextSchedulePlan(100, self.max_batch_size_full)

        avg = sum(self.observed_percentages) / len(self.observed_percentages) if self.observed_percentages else 0
        self.prefill_estimate_time = (int(self.prefill_estimate_time[0] * (1 + avg + 0.2)), round(avg,2))
        self.pause = (self.prefill_estimate_time[0] > self.llm_slo_info['first'] or
            processing_pred_latency * (1 + avg + 0.1) > self.llm_slo_info['speed']
        )

        if self.solo_mode: # First round back from solo_mode;
            self.solo_mode = False
            return ContextSchedulePlan(self.pre_solo_mps, self.max_batch_size if self.pause != True else 0)

        if(self.pause or not self.current_plan):
            return ContextSchedulePlan(current_mps, self.max_batch_size if self.pause != True else 0)

        # Normal mode: check if interval has elapsed for MPS adjustment
        if time.perf_counter() - self.last_context_switch_time > self.context_switch_interval:
            new_mps = current_mps
            if avg < 0.15:
                new_mps = min(100, new_mps + self.MPS_step)
            elif avg > 0.25:
                new_mps = max(10, new_mps - self.MPS_step)
            if new_mps != self.current_plan.mps:
                plan = ContextSchedulePlan(new_mps, self.max_batch_size)
                self.last_context_switch_time = time.perf_counter()
                self.observed_percentages.clear()
                return plan
        return ContextSchedulePlan(current_mps, self.max_batch_size)

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()
        if(intensity_data.processing):
            intensity_data.batch_data.append(intensity_data.processing)
        pred_latency = [float(x) for x in context_predict(self.predictor, intensity_data)]
        log_processing = None
        if(intensity_data.processing):
            log_processing = list(intensity_data.processing.pred_feature_data()) + [intensity_data.processing.latency, pred_latency[-1]]
        plan = self.get_scheduling_plan(intensity_data, pred_latency)

        current_timestamp = round(time.perf_counter() - self.model_client.runtime_data.start_time_pointer, 3)
        self.execution_logs.append([current_timestamp, 'max' if plan.batch_size else 0, plan.mps,
            list(zip([x.latency for x in intensity_data.batch_data], pred_latency)),
            log_processing, (intensity_data.NQ, intensity_data.waiting_prompt_tkns, self.prefill_estimate_time)])
        self.execute(plan)

    def execute(self, plan):
        if self.current_plan is None or plan.mps != self.current_plan.mps or plan.batch_size != self.current_plan.batch_size:
            self.model_client.broadcast_workers(plan, sync=SyncType.ALL)
            self.current_plan = plan
        time.sleep(self.time_interval)

    def _serve(self):
        self.scheduler_init()
        print("Execution speed", self.max_process_speed)
        # May have remaining data in intensity server; fetch once to clear buffer
        current_intensity = get_llm_intensity_unix(self.data_format)
        self.start_serve()

        time_limit = self.model_client.runtime_data.time_limit
        test_time = time_limit - time.perf_counter()
        with tqdm(total=test_time, unit="s", bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]", disable = True) as pbar:
            last = time.perf_counter()
            while time.perf_counter() < time_limit:
                current_intensity = get_llm_intensity_unix(self.data_format)
                self.schedule(current_intensity)
                now = time.perf_counter()
                pbar.update(now - last)
                last = now
        self.model_client.broadcast_workers(ContextSchedulePlan(-1, -1, True), sync = SyncType.ALL)

        self.model_client.write_csv("batch_predicted.csv",
            ["Scheduled Tkns", "Cached Tkns", "Real Latency", "Pred Latency"],
            self.prediction_logs)
        self.model_client.write_csv("batch_execution.csv",
            ["Time", "Batch size", "MPS", "Details", "Processing", "Additional_data"],
            self.execution_logs
        )
        mps_data = TraceAxisData(TracePlotData('MPS %', [(x[0], x[2]) for x in self.execution_logs]))
        plot_trace_graph(self.model_client.get_file_path('mps_percentage.png'), mps_data)
