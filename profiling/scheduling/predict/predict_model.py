import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import time
from sklearn.preprocessing import StandardScaler
from joblib import load

class PredictModel():
    def __init__(self, tag):
        self.path_prefix = "/home/k43015/MPSTest/data/predict/predict_model/results/BurstGPT2_100_300_400000_6"
        if(len(self.path_prefix) != 0):
            self.path_prefix += "/"
        self.get_saved_model_path(tag)
        self.time_test_repeats = 10
        self.load_sc()
    
    def load_sc(self):
        self.sc = StandardScaler()
        self.sc=load(self.path_prefix + 'std_scaler.bin')

    def get_saved_model_path(self, tag):
        self.save_model_path =  self.path_prefix + "model_" + self.model_name + "_" + tag + "." + self.saved_model_suffix_name

    def train(self, X_data, y_data, train_config = {}):
        pass

    def eval(self, X_data, y_data = None):
        pass

class SKLPredictModel(PredictModel):
    def train(self, X_data, y_data, train_config = {}):
        self.model.fit(X_data, y_data)
        _, test_loss = self.eval(X_data, y_data)
        print(f"📉 Final Train Loss (MSE): {test_loss:.4f}")

    def eval(self, X_data, y_data = None):
        test_outputs = self.model.predict(X_data)
        mse = None
        if isinstance(y_data, np.ndarray):
            mse = mean_squared_error(test_outputs, y_data)
        start_time = time.time()
        for i in range(self.time_test_repeats):
            _ = self.model.predict(X_data[i:i+1, :])
        end_time = time.time()
        # r2 = r2_score(test_outputs, y_data)
        return test_outputs, mse, (end_time - start_time) / self.time_test_repeats

    def save(self):
        print("Saving to:", self.save_model_path)
        with open(self.save_model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)
    
    def load(self):
        if os.path.exists(self.save_model_path):
            print("Loading from:", self.save_model_path)
            with open(self.save_model_path, "rb") as model_file:
                self.model = pickle.load(model_file)

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
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(self.time_test_repeats):
            X_tensor = torch.from_numpy(X_data[i:i+1, :]).to(self.device)
            _ = self.model(X_tensor)
            torch.cuda.synchronize()
        end_time = time.time()
        return test_outputs.cpu().numpy(), test_loss.item(), (end_time - start_time) / self.time_test_repeats
    
    def save(self):
        print("Saving to:", self.save_model_path)
        torch.save(self.model.state_dict(), self.save_model_path)

    def load(self):
        if os.path.exists(self.save_model_path):
            print("Loading from:", self.save_model_path)
            self.model.load_state_dict(torch.load(self.save_model_path, weights_only=True))
            self.model.to(self.device)  # Ensure the model is also moved to the GPU
    
    def exec(self, X_data):
        X_data_std = self.sc.transform(X_data)
        tensor = torch.tensor(X_data_std, dtype=torch.float32).to("cuda")
        return self.model(tensor)