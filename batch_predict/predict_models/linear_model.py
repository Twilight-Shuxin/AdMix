from sklearn.linear_model import LinearRegression
from predict_models.predict_model import SKLPredictModel

class LinearPredictModel(SKLPredictModel):
    def __init__(self):
        self.model_name = "linear"
        self.model = LinearRegression()
        super().__init__()
