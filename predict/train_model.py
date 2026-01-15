import torch
import torch.nn as nn
import os
import sys
from utils import get_model

# python train_model.py MLP prefill
# python train_model.py MLP

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
    model_type = sys.argv[1]
    predict_name_list = ["prefill", "decode"]
    if(len(sys.argv) > 2):
        predict_name_list = [sys.argv[2]]

    for predict_name in predict_name_list:
        print("Training for:", predict_name, "...")
        X_train_tensor, y_train_tensor = torch.load(predict_name + "_train.pt", weights_only = True)
        X_test_tensor, y_test_tensor = torch.load(predict_name + "_validate.pt", weights_only = True)

        X_train_np = X_train_tensor.numpy()
        y_train_np = y_train_tensor.numpy()
        X_test_np = X_test_tensor.numpy()
        y_test_np = y_test_tensor.numpy()
        input_dim = X_train_tensor.shape[1]

        saved_model_path = 'model_' + predict_name + '_' + model_type + '.pkl'
        get_model(model_name, input_dim, device)
        model.load()
        model.train(X_train_np, y_train_np)
        model.save()
        _, test_loss, avg_time = model.eval(X_test_np, y_test_np)
        print(f"📉 Final Validation Loss (MSE): {test_loss:.4f}, avg_time: {avg_time}")

if __name__ == "__main__":
    main()