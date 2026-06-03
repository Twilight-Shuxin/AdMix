from admix.profiling.scheduling import *

_SCHEDULERS = {
    "context": ContextScheduler,
    "contextminus": ContextMinusScheduler,
}

def get_scheduler_class(scheduler_name):
    if scheduler_name is None:
        return None
    if scheduler_name.startswith("constant"):
        return ConstantScheduler
    if scheduler_name.startswith("pconstant"):
        return PConstantScheduler
    return _SCHEDULERS.get(scheduler_name)


def get_scheduler(scheduler_name):
    if scheduler_name is None:
        return None
    scheduler_class = get_scheduler_class(scheduler_name)
    if scheduler_name.startswith("constant") or scheduler_name.startswith("pconstant"):
        if "[" in scheduler_name:
            batch_size = int(scheduler_name[scheduler_name.find('[') + 1 : scheduler_name.find(']')])
            return scheduler_class(batch_size)
    return scheduler_class()