from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from predict_model import SKLPredictModel
import pickle
import os
import numpy as np

class LinearPredictModel(SKLPredictModel):
    def __init__(self, tag):
        self.model_name = "linear"
        self.saved_model_suffix_name = "pkl"
        self.model = LinearRegression()
        super().__init__(tag)
