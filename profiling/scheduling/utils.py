from profiling.scheduling.AIMD_scheduler import AIMDScheduler
from profiling.scheduling.advanced_AIMD_scheduler import AdvancedAIMDScheduler
from profiling.scheduling.predictive_scheduler import PredictiveScheduler

schedulers = {
    "AIMD": AIMDScheduler,
    "advanced_AIMD": AdvancedAIMDScheduler,
    "predictive": PredictiveScheduler
}