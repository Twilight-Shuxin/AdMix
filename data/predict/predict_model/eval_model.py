import pandas as pd
import torch
import torch.nn as nn
from MLP_model import MLPPredictModel
from linear_model import LinearPredictModel
from poisson_model import PoissonPredictModel
from decision_tree_model import DecisionTreePredictModel
from random_forest_model import RandomForestModel
from plot import plot, plot_abs_error_cdf, plot_abs_pct_error_cdf
import sys

# python eval_model.py linear
# python eval_model.py MLP
# python eval_model.py linear data.csv

pd.set_option('display.max_columns', None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability

predict_list = ["prefill", "decode"]
model_type = sys.argv[1]
df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else 'test_data_saved.csv')
test_results = {}

for predict_name in predict_list:
    X_test_tensor, y_test_tensor = torch.load(predict_name + "_test.pt", weights_only=True)
    X_test_np = X_test_tensor.numpy()
    y_test_np = y_test_tensor.numpy()
    input_dim = X_test_tensor.shape[1]
    if(model_type == "MLP"):
        model = MLPPredictModel(predict_name, input_dim, device)
    elif (model_type == "linear"):
        model = LinearPredictModel(predict_name)
    elif (model_type == "poisson"):
        model = PoissonPredictModel(predict_name)
    elif (model_type == "decision_tree"):
        model = DecisionTreePredictModel(predict_name)
    elif (model_type == "random_forest"):
        model = RandomForestModel(predict_name)
    model.load()
    test_result, test_loss, avg_time = model.eval(X_test_np, y_test_np)
    test_results[predict_name] = [test_result, test_loss]
    df['Predicted ' + predict_name] = test_result
    print(f"📉 Test Loss (MSE) {predict_name}: {test_loss:.4f}, process time (s):", avg_time)
    plot(df['Timestamp (s)'].values, y_test_np, test_result, predict_name + '_predicted_' + model_type)
    plot_abs_error_cdf(y_test_np, test_result, predict_name + '_abs_error_' + model_type)

# Save to new CSV file
df.to_csv('predicted_data_' + model_type + '.csv', index=False)
df_out = df[
    [
        "Timestamp (s)",
        "t1 prefill average length",
        "Prefill throughput",
        "Predicted prefill",
        "Decode throughput",
        "Predicted decode",
    ]
].rename(
    columns={
        "Timestamp (s)": "Time",
        "t1 prefill average length": "Batch Size",
        "Prefill throughput": "Prefill_Tp",
        "Predicted prefill": "Predicted_Prefill_Tp",
        "Decode throughput": "Decode_Tp",
        "Predicted decode": "Predicted_Decode_Tp",
    }
)
df.to_csv('predicted_data_' + model_type + '_simplified.csv', index=False)