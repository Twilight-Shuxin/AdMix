import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class PredictModel():
    def get_saved_model_name(self, tag):
        self.save_model_name =  "model_" + self.model_name + "_" + tag + "." + self.saved_model_suffix_name

    def train(self, X_data, y_data, train_config = {}):
        pass

    def eval(self, X_data, y_data = None):
        pass
    
    def save_model(self):
        pass
    
    def load_model(self):
        pass

class PytorchPredictModel(PredictModel):
    def train(self, X_data, y_data, train_config = {}):
        num_epochs = train_config.get("num_epochs", 1000) 
        batch_size = train_config.get("batch_size", 32)
        X_train_tensor = torch.from_numpy(X_data).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_data).float().to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.model.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.model.criterion(outputs, batch_y.view(-1, 1))
                loss.backward()
                self.model.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            # Print average loss for the epoch
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(train_loader.dataset)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def eval(self, X_data, y_data = None):
        X_test_tensor = torch.from_numpy(X_data).float().to(self.device)
        if isinstance(y_data, np.ndarray):
            y_test_tensor = torch.from_numpy(y_data).float().to(self.device)
        test_loss = None
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            test_outputs = test_outputs.squeeze()
            if isinstance(y_data, np.ndarray):
                test_loss = self.model.criterion(test_outputs, y_test_tensor)
        return test_outputs.cpu().numpy(), test_loss.item()
    
    def save(self):
        torch.save(self.model.state_dict(), self.save_model_name)

    def load(self):
        if os.path.exists(self.save_model_name):
            self.model.load_state_dict(torch.load(self.save_model_name, weights_only=True))
            self.model.to(self.device)  # Ensure the model is also moved to the GPU