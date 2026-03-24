from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from predict_models.predict_model import SKLPredictModel

class PolynomialPredictModel(SKLPredictModel):
    def __init__(self, degree=2):
        self.model_name = f"polynomial_deg{degree}"
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ])
        super().__init__()
