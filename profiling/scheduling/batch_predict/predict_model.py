
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import time
from joblib import load

GBATCH_PREDICT_MODEL_FOLDER = Path(os.getenv("GBATCH_PREDICT_MODEL_PATH", os.path.dirname(os.path.abspath(__file__))))

class WrappedStandardScaler():
    def __init__(self, model_folder):
        self.save_path = model_folder / "std_scaler.bin"

    def transform(self, x):
        return self.sc.transform(x)

    def load(self):
        if os.path.exists(self.save_path):
            print("Loaded scaler from", self.save_path)
            self.sc = load(self.save_path)

class PredictModel():
    def __init__(self, llm_model):
        self.model_folder = GBATCH_PREDICT_MODEL_FOLDER / "models" / llm_model
        self.set_model_path()
        self.sc = WrappedStandardScaler(self.model_folder)
        self.load()
        self.model.eval()

    def set_model_path(self):
        model_filename = f"model_{self.model_name}.{self.saved_model_suffix_name}"
        self.load_model_path = self.model_folder / model_filename

    def load(self):
        self._load()
        self.sc.load()

class SKLPredictModel(PredictModel):
    def __init__(self, llm_model):
        self.saved_model_suffix_name = "pkl"
        super().__init__(llm_model)

    def exec(self, X_data):
        return self.model.predict(self.sc.transform(X_data))

    def _load(self):
        if os.path.exists(self.load_model_path):
            with open(self.load_model_path, "rb") as model_file:
                self.model = pickle.load(model_file)

class PytorchPredictModel(PredictModel):
    def __init__(self, llm_model):
        self.saved_model_suffix_name = "pth"
        super().__init__(llm_model)

    def exec(self, X_data):
        X_test_tensor = torch.from_numpy(np.asarray(self.sc.transform(X_data))).float().to(self.device)
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor).squeeze(dim=1)
            test_outputs = torch.round(test_outputs).clamp(min=0)
        return test_outputs.cpu().numpy()

    def _load(self):
        if os.path.exists(self.load_model_path):
            self.model.load_state_dict(torch.load(self.load_model_path, weights_only=True))
            print("Loaded model from", self.load_model_path)
            self.model.to(self.device)  # Ensure the model is also moved to the GPU