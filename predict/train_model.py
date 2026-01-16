import torch
import torch.nn as nn
import os
import sys
from utils import get_model, load_data
from predict_models.constants import DEFAULT_PRED_LIST, TRAIN_DATA_FILE, VALIDATE_DATA_FILE

# python train_model.py MLP prefill
# python train_model.py MLP

def main():
    model_type = sys.argv[1]
    if(len(sys.argv) > 2):
        predict_name_list = [sys.argv[2]]
    else:
        predict_name_list = DEFAULT_PRED_LIST

    train_data = load_data(TRAIN_DATA_FILE)
    validate_data = load_data(VALIDATE_DATA_FILE)

    for predict_name in predict_name_list:
        print("Training for:", predict_name, "...")

        input_dim = train_data.X.shape[1]
        model = get_model(model_type, predict_name, input_dim)
        model.load()
        model.train(train_data.X, train_data.y[predict_name])
        model.save()
        _, validation_loss, avg_time = model.eval(validate_data.X, validate_data.y[predict_name])
        print(f"📉 Final Validation Loss (MSE): {validation_loss:.4f}, avg_time: {avg_time}")

if __name__ == "__main__":
    main()