__all__ = [
    "ConstantScheduler",
    "ContextScheduler",
    "ContextMinusScheduler",
    "PConstantScheduler",
]

def __getattr__(name):
    match name:
        case "ConstantScheduler":
            from .constant_scheduler import ConstantScheduler
            return ConstantScheduler
        case "ContextScheduler":
            from .context_scheduler import ContextScheduler
            return ContextScheduler
        case "ContextMinusScheduler":
            from .contextminus_scheduler import ContextMinusScheduler
            return ContextMinusScheduler
        case "PConstantScheduler":
            from .pconstant_scheduler import PConstantScheduler
            return PConstantScheduler
        case "get_scheduler_class":
            from .utils import get_scheduler_class
            return get_scheduler_class
    raise AttributeError(f"module {__name__} has no attribute {name}")