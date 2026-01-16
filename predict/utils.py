import torch
from predict_models import *
from predict_models.constants import SAVED_SCALER_FILE

def get_model(model_type, predict_name, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match model_type:
        case "MLP":
            return MLPPredictModel(predict_name, input_dim, device)
        case "linear":
            return LinearPredictModel(predict_name)
        case "poisson":
            return PoissonPredictModel(predict_name)
        case "decision_tree":
            return DecisionTreePredictModel(predict_name)
        case "random_forest":
            return RandomForestModel(predict_name)

class PackedData():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.input_dim = X.shape[1]

def load_data(path):
    X, y_prefill, y_decode = torch.load(path, weights_only = True)
    return PackedData(X.numpy(), {
            "prefill":y_prefill.numpy(), 
            "decode":y_decode.numpy()})