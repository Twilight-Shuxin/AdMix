import time
from typing import List
from profiling.models.workers.constants import SyncType
from profiling.scheduling.scheduler import SchedulingDataFormat
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.batch_base_scheduler import BatchBaseScheduler, BatchSchedulingDataFormat

class BatchScheduler(BatchBaseScheduler):
    data_format = BatchSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def __init__(self):
        super().__init__()
        self.solo_mode = False
        self.time_interval = 0.05 # 50ms
    
    def get_action(self, intensity_data, pred_latency):
        if(intensity_data.processing):
            processed_latency, processing_pred_latency = intensity_data.batch_data.pop().latency, pred_latency.pop()
            processing = (processed_latency - processing_pred_latency) / processed_latency if processed_latency > processing_pred_latency else 0
        else: 
            processing = 0

        real_latency = [x.latency for x in intensity_data.batch_data]
        percentages = [(r - p) / p for r, p in zip(real_latency, pred_latency) if r - 3 > p and p != 0]
        # Log predictions
        if intensity_data.batch_data:
            self.prediction_logs.extend([
                feature.copy() + [latency, pred_lat]
                for feature, latency, pred_lat in zip(
                    intensity_data.pred_feature_data(), real_latency, pred_latency,
                )
            ])
        
        interference = False
        if (not intensity_data.batch_data) and intensity_data.NQ == 0: # No running queries — run solo
            if(not self.solo_mode):
                self.batch_size = 1 if self.batch_size == 0 else self.batch_size
                self.solo_mode = True
            else:
                self.batch_size = min(self.batch_size * 2, self.max_batch_size)
            self.solo_time_sum += self.time_interval
            return self.Action.FLAT
        
        # LLM queries running
        if self.solo_mode: # First round back from solo mode — reset
            self.solo_mode = False
            return self.Action.DOWN
        self.solo_mode = False

        max_pct = max(percentages) if percentages else 0
        avg_pct = sum(percentages) / len(percentages) if percentages else 0
        prefill_estimate_time = intensity_data.waiting_prompt_tkns * self.max_process_speed
        # Interference is detected if 
        #   1. Running queries exceeds a certain threshold
        #   2. The delayed percentage exceeds pre-defined limits
        #   3. Waiting prefill tokens are large
        interference = (processing > 0.1 or max_pct > 0.5 or avg_pct > 0.2 or prefill_estimate_time > 0.6 * self.llm_slo_info['first'])
        if(interference):
            return self.Action.DOWN
        if(avg_pct < 0.15):
            return self.Action.UP
        return self.Action.FLAT

    def schedule(self, intensity_data):
        self.start_round_time = time.perf_counter()
        if(intensity_data.processing):
            intensity_data.batch_data.append(intensity_data.processing)
        pred_latency = [float(x) for x in self.get_prediction(intensity_data)]
        log_processing = None
        if(intensity_data.processing):
            log_processing = list(intensity_data.processing.pred_feature_data()) + [intensity_data.processing.latency, pred_latency[-1]]
        action = self.get_action(intensity_data, pred_latency)
        self.execute_action(action)
        
        current_timestamp = round(time.perf_counter() - self.model_client.runtime_data.start_time_pointer, 3)
        self.execution_logs.append([current_timestamp, self.batch_size, 
            list(zip([x.latency for x in intensity_data.batch_data], pred_latency)), 
            log_processing, (intensity_data.NQ, intensity_data.waiting_prompt_tkns)])
        self.execute()
    
    def execute(self):
        end_round_time = self.start_round_time + self.time_interval
        self.batch_size = self.max_batch_size
        schedule_plan = BasicSchedulePlan(self.batch_size)
        self.last_execution_time = 0
        time.sleep(self.time_interval)
        return
        while(time.perf_counter() + self.last_execution_time < end_round_time):
            start_time = time.perf_counter()
            self.model_client.broadcast_workers(schedule_plan, sync = SyncType.ALL)
            self.last_execution_time = time.perf_counter() - start_time