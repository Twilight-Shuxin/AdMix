__all__ = [
    "ConstantScheduler",
    "AIMDScheduler",
    "PredictiveScheduler",
    "BatchScheduler",
    "AIMDSchedulingDataFormat",
    "PredictiveSchedulingDataFormat"
]

def __getattr__(name):
    match name:
        case "ConstantScheduler":
            from .constant_scheduler import ConstantScheduler
            return ConstantScheduler
        case "AIMDScheduler":
            from .AIMD_scheduler import AIMDScheduler
            return AIMDScheduler
        case "PredictiveScheduler":
            from .predictive_scheduler import PredictiveScheduler
            return PredictiveScheduler
        case "BatchScheduler":
            from .batch_scheduler import BatchScheduler
            return BatchScheduler
        case "AIMDSchedulingDataFormat":
            from .AIMD_scheduler import AIMDSchedulingDataFormat
            return AIMDSchedulingDataFormat
        case "PredictiveSchedulingDataFormat":
            from .predictive_scheduler import PredictiveSchedulingDataFormat
            return PredictiveSchedulingDataFormat
        case "get_scheduler_class":
            from .utils import get_scheduler_class
            return get_scheduler_class
    raise AttributeError(f"module {__name__} has no attribute {name}")