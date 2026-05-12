import time
from admix.profiling.models.workers.constants import SyncType
from admix.profiling.scheduling.schedule_plan import BasicSchedulePlan
from admix.profiling.scheduling.batch_base_scheduler import (
    BatchBaseScheduler, BatchSchedulingDataFormat, ObservationData, ObservationSingleBatchData
)
from admix.profiling.scheduling.monitor import get_llm_intensity_unix
from admix.profiling.scheduling.batch_predict.MLP_model import MLPPredictModel

class PConstantScheduler(BatchBaseScheduler):
    data_format = BatchSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.observed_percentages = []
        self.pause = False
        self.prefill_estimate_time = (0, 0)
        self.current_plan = None
        self.time_interval = 0.05  # 50ms
    
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

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()
        if intensity_data.processing:
            intensity_data.batch_data.append(intensity_data.processing)
        pred_latency = [float(x) for x in self.get_prediction(intensity_data)]

        processing_pred_latency = 0
        self.prefill_estimate_time = (int(intensity_data.waiting_prompt_tkns * self.max_process_speed), 0)
        if intensity_data.processing:
            _, processing_pred_latency = intensity_data.batch_data.pop().latency, pred_latency.pop()
        real_latency = [x.latency for x in intensity_data.batch_data]

        if not self.pause:
            self.observed_percentages.extend(
                [(r - p) / p if r - 3 > p and p > 0 else 0 for r, p in zip(real_latency, pred_latency)]
            )
            if len(self.observed_percentages) > 20:
                self.observed_percentages = self.observed_percentages[-20:]

        if intensity_data.batch_data:
            self.prediction_logs.extend([
                feature.copy() + [latency, pred_lat]
                for feature, latency, pred_lat in zip(
                    intensity_data.pred_feature_data(), real_latency, pred_latency,
                )
            ])

        avg = sum(self.observed_percentages) / len(self.observed_percentages) if self.observed_percentages else 0
        self.prefill_estimate_time = (int(self.prefill_estimate_time[0] * (1 + avg + 0.2)), round(avg, 2))
        self.pause = (
            self.prefill_estimate_time[0] > self.llm_slo_info['first'] or
            processing_pred_latency * (1 + avg + 0.1) > self.llm_slo_info['speed']
        )

        current_timestamp = round(time.perf_counter() - self.model_client.runtime_data.start_time_pointer, 3)
        self.execution_logs.append([
            current_timestamp,
            'paused' if self.pause else self.batch_size,
            list(zip([x.latency for x in intensity_data.batch_data], pred_latency)),
            None,
            (intensity_data.NQ, intensity_data.waiting_prompt_tkns, self.prefill_estimate_time),
        ])

        if self.pause:
            return BasicSchedulePlan(0)
        return BasicSchedulePlan(self.batch_size)

    def execute(self, plan):
        end_round_time = self.start_round_time + self.time_interval
        self.last_execution_time = 0
        if plan.batch_size != 0:
            while time.perf_counter() + self.last_execution_time < end_round_time:
                start_time = time.perf_counter()
                self.model_client.broadcast_workers(plan, sync=SyncType.ALL)
                self.last_execution_time = time.perf_counter() - start_time
        left_time = end_round_time - time.perf_counter()
        if left_time > 0.001:
            time.sleep(left_time)

    def _serve(self):
        self.scheduler_init()
        print("Execution speed", self.max_process_speed)
        current_intensity = get_llm_intensity_unix(self.data_format)
        self.start_serve()
        while time.perf_counter() < self.model_client.runtime_data.time_limit:
            current_intensity = get_llm_intensity_unix(self.data_format)
            self.execute(self.schedule(current_intensity))
        self.model_client.write_csv("batch_predicted.csv",
            ["Scheduled Tkns", "Cached Tkns", "Real Latency", "Pred Latency"],
            self.prediction_logs)
        self.model_client.write_csv("batch_execution.csv",
            ["Time", "Batch size", "Details", "Processing", "Additional_data"],
            self.execution_logs)
