import torch.nn as nn
import torch.optim as optim
from predict_models.predict_model import PytorchPredictModel

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, device):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1),
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

class MLPPredictModel(PytorchPredictModel):
    def __init__(self, input_dim, device):
        self.model = MLPRegressor(input_dim, device)
        self.model_name = "MLP"
        self.saved_model_suffix_name = "pth"
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        super().__init__()