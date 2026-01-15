import torch
from predict_models import *

def get_model(model_type, input_dim):
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

"""
    Load data from .pt files and return data as numpy arrays along with its input dim.
"""
def load_data(predict_name, data_class):
    X_tensor, y_tensor = torch.load(predict_name + "_" + data_class + ".pt", weights_only=True)
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()
    input_dim = X_tensor.shape[1]
    return input_dim, X_np, y_np

def load_scaler():
    sc = StandardScaler()
    model_path = os.getenv("MODEL_PATH", "outputs") + "/"
    sc = load(model_path + 'std_scaler.bin')
    return sc