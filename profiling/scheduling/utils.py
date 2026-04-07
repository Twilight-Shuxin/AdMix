from profiling.scheduling import *

def get_scheduler_class(scheduler_name):
    if(scheduler_name is None):
        return None
    if(scheduler_name.startswith("constant")):
        return ConstantScheduler
    match(scheduler_name):
        case "AIMD":
            return AIMDScheduler
        case "predictive":
            return PredictiveScheduler
        case "batch":
            return BatchScheduler
        case "context":
            return ContextScheduler
    return None  

def get_scheduler(scheduler_name):
    if scheduler_name is None:
        return None
    scheduler_class = get_scheduler_class(scheduler_name)
    if(scheduler_name.startswith("constant")):
        if("[" in scheduler_name):
            batch_size = int(scheduler_name[scheduler_name.find('[') + 1 : scheduler_name.find(']')])
            return scheduler_class(batch_size)
    return scheduler_class()