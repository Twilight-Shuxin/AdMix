import time
from admix.profiling.scheduling.context_scheduler import ContextScheduler
from admix.profiling.scheduling.schedule_plan import ContextSchedulePlan


class ContextMinusScheduler(ContextScheduler):
    """ContextScheduler without the proactive pause mechanism."""

    def get_scheduling_plan(self, intensity_data, pred_latency):
        current_mps = self.current_plan.mps if self.current_plan else self.normal_plan.mps

        self.prefill_estimate_time = (int(intensity_data.waiting_prompt_tkns * self.max_process_speed), 0)
        if intensity_data.processing:
            _, _ = intensity_data.batch_data.pop().latency, pred_latency.pop()
        real_latency = [x.latency for x in intensity_data.batch_data]
        if(not self.solo_mode):
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

        if intensity_data.NQ == 0:
            if not self.solo_mode:
                self.pre_solo_mps = current_mps
                self.solo_mode = True
            return ContextSchedulePlan(100, self.max_batch_size_full)

        avg = sum(self.observed_percentages) / len(self.observed_percentages) if self.observed_percentages else 0
        self.prefill_estimate_time = (int(self.prefill_estimate_time[0] * (1 + avg + 0.2)), round(avg, 2))

        if self.solo_mode:
            self.solo_mode = False
            return ContextSchedulePlan(self.pre_solo_mps, self.max_batch_size)

        if not self.current_plan:
            return ContextSchedulePlan(current_mps, self.max_batch_size)

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
