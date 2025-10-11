from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
from predict_model import SKLPredictModel
import pickle
import os
import numpy as np

class PoissonPredictModel(SKLPredictModel):
    def __init__(self, tag):
        self.model_name = "poisson"
        self.saved_model_suffix_name = "pkl"
        self.model = PoissonRegressor()
        super().__init__(tag)
