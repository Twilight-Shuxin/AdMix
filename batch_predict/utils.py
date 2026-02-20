import torch
from predict_models import *

def get_model(model_type, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match model_type:
        case "MLP":
            return MLPPredictModel(input_dim, device)
        case "linear":
            return LinearPredictModel()
        case "poisson":
            return PoissonPredictModel()
        case "decision_tree":
            return DecisionTreePredictModel()
        case "random_forest":
            return RandomForestModel()

class PackedData():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.input_dim = X.shape[1]

def load_data(path):
    X, y = torch.load(path, weights_only = True)
    return PackedData(X.numpy(), y.numpy())