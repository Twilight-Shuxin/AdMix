from pydantic import BaseModel
SCHEDULING_BIN_SIZE = 0.1

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