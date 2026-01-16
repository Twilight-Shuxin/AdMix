
import os
import pickle
import numpy as np
import torch
import time
from joblib import dump, load
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

SAVED_SCALER_FILE = "std_scaler.bin"

class WrappedStandardScaler():
    def transform(self, x):
        return self.sc.transform(x)
    
    def load(self):
        self.sc = load(SAVED_SCALER_FILE)
        
class PredictModel():
    def __init__(self, tag):
        self.set_model_path(tag)
        self.sc = WrappedStandardScaler()
        self.loaded = False

    def set_model_path(self, tag):
        model_filename = f"model_{self.model_name}_{tag}.{self.saved_model_suffix_name}"
        self.load_model_path = model_filename

    def load(self):
        self.loaded = self._load()

class SKLPredictModel(PredictModel):
    def __init__(self, tag):
        self.saved_model_suffix_name = "pkl"
        super().__init__(tag)
    
    def exec(self, X_data):
        return self.model.predict(X_data)

    def _load(self):
        if os.path.exists(self.load_model_path):
            print("Loading from:", self.load_model_path)
            with open(self.load_model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
            self.sc.load()
            return True
        return False

class PytorchPredictModel(PredictModel):
    def __init__(self, tag):
        self.model.eval()
        super().__init__(tag)

    def exec(self, X_data):
        X_test_tensor = torch.from_numpy(np.asarray(X_data)).float().to(self.device)
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor).squeeze()
            test_outputs = torch.round(test_outputs).clamp(min=0)
        return test_outputs.cpu().numpy()

    def _load(self):
        if os.path.exists(self.load_model_path):
            print("Loading from:", self.load_model_path)
            self.model.load_state_dict(torch.load(self.load_model_path, weights_only=True))
            self.model.to(self.device)  # Ensure the model is also moved to the GPU
            return True
        return False