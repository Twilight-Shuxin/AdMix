from pydantic import BaseModel
from profiling.scheduling.AIMD_scheduler import AIMDScheduler
from profiling.scheduling.advanced_AIMD_scheduler import AdvancedAIMDScheduler
from profiling.scheduling.predictive_scheduler import PredictiveScheduler
from profiling.scheduling.constant_scheduler import ConstantScheduler

def get_scheduler_class(scheduler_name):
    if(scheduler_name is None):
        return None
    if(scheduler_name.startswith("constant")):
        return ConstantScheduler
    match(scheduler_name):
        case "AIMD":
            return AIMDScheduler
        case "advanced_AIMD":
            return AdvancedAIMDScheduler
        case "predictive":
            return PredictiveScheduler
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

class FeatureModel(BaseModel):
    def vector(self):
        vec = []
        for name, field in self.model_fields.items():
            value = getattr(self, name)
            if isinstance(value, FeatureModel):
                vec.extend(value.feature_vector())
            else:
                vec.append(value)
        return vec