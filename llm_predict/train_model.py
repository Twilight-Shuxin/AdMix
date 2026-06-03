import torch
import torch.nn as nn
import os
import sys
from utils import get_model, load_data
from predict_models.constants import TRAIN_DATA_FILE, VALIDATE_DATA_FILE

# python train_model.py MLP

def main():
    model_type = sys.argv[1]
    train_data = load_data(TRAIN_DATA_FILE)
    validate_data = load_data(VALIDATE_DATA_FILE)

    input_dim = train_data.X.shape[1]
    model = get_model(model_type, input_dim)
    model.load()
    model.train(train_data.X, train_data.y)
    model.save()
    _, validation_loss, avg_time = model.eval(validate_data.X, validate_data.y)
    print(f"📉 Final Validation Loss (MSE): {validation_loss:.4f}, avg_time: {avg_time}")

if __name__ == "__main__":
    main()