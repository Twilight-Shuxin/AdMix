from sklearn.linear_model import LinearRegression
from predict_model import SKLPredictModel

class LinearPredictModel(SKLPredictModel):
    def __init__(self, tag):
        self.model_name = "linear"
        self.saved_model_suffix_name = "pkl"
        self.model = LinearRegression()
        super().__init__(tag)
