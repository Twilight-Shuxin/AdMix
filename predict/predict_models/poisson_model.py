from sklearn.linear_model import PoissonRegressor
from predict_model import SKLPredictModel

class PoissonPredictModel(SKLPredictModel):
    def __init__(self, tag):
        self.model_name = "poisson"
        self.saved_model_suffix_name = "pkl"
        self.model = PoissonRegressor()
        super().__init__(tag)
