import time
from typing import List
from profiling.models.workers.constants import SyncType
from profiling.scheduling.scheduler import SchedulingDataFormat
from profiling.scheduling.schedule_plan import BasicSchedulePlan
from profiling.scheduling.batch_base_scheduler import BatchBaseScheduler, BatchSchedulingDataFormat

class BatchScheduler(BatchBaseScheduler):
    data_format = BatchSchedulingDataFormat
    scheduling_plan = BasicSchedulePlan

    def get_action(self, intensity_data, pred_latency):
        # No batch information has been collected in the last scheduling interval
        if(not intensity_data.batch_data):
            # If no queries are under execution, increase batch size. 
            if(intensity_data.NQ == 0):
                return self.Action.UP
            # Else reduce batch size
            # TODO: Improve proactive method to reduce batch size
            return self.Action.DOWN

        real_latency = [x.latency for x in intensity_data.batch_data]
        percentages = [(r - p) / p for p, r in zip(pred_latency, real_latency) if r > p and p != 0]

        self.prediction_logs.extend([
                feature.copy() + [latency, pred_latency]
                for feature, latency, pred_latency in zip(
                    intensity_data.pred_feature_data(), real_latency, pred_latency,
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