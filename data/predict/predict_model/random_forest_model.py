from sklearn.ensemble import RandomForestRegressor
from predict_model import SKLPredictModel
import os

class RandomForestModel(SKLPredictModel):
    def __init__(self, tag):
        self.model_name = "random_forest"
        self.saved_model_suffix_name = "pkl"
        n_estimators = 100
        random_state = 42
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        super().__init__(tag)