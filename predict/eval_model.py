import pandas as pd
from constants import SAVED_TEST_OUTPUT_FILE
from utils import get_model, load_data
import sys

# python eval_model.py linear

def main():
    pd.set_option('display.max_columns', None)
    predict_list = ["prefill", "decode"]
    model_type = sys.argv[1]
    df = pd.read_csv(SAVED_TEST_OUTPUT_FILE)
    test_results = {}

    for predict_name in predict_list:
        input_dim, X_test_np, y_test_np = load_data(predict_name, 'test')
        model = get_model(model_type, input_dim)
        model.load()
        test_result, test_loss, avg_time = model.eval(X_test_np, y_test_np)

        test_results[predict_name] = [test_result, test_loss]
        df['Predicted ' + predict_name] = test_result

        print(f"📉 Test Loss (MSE) {predict_name}: {test_loss:.4f}, process time (s):", avg_time)
        
        plot_predicted(df['Timestamp (s)'].values, y_test_np, test_result, predict_name + '_predicted_' + model_type)
        plot_abs_error_cdf(y_test_np, test_result, predict_name + '_abs_error_' + model_type)

    # Save to new CSV file
    df.to_csv('predicted_' + model_type + '.csv', index=False)

if __name__ == "__main__":
    main()