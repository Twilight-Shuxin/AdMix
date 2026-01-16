import sys
import pandas as pd
from utils import get_model, load_data
from plot import plot_predicted, plot_abs_error_cdf
from predict_models.constants import SAVED_TEST_DATA_CSV_FILE, DEFAULT_PRED_LIST, TEST_DATA_FILE, PredHeaders, OUTPUT_FOLDER

# python eval_model.py linear

def main():
    model_type = sys.argv[1]
    predict_list = DEFAULT_PRED_LIST

    df = pd.read_csv(SAVED_TEST_DATA_CSV_FILE)
    data = load_data(TEST_DATA_FILE)

    for predict_name in predict_list:
        model = get_model(model_type, predict_name, data.input_dim)
        model.load()
        test_result, test_loss, avg_time = model.eval(data.X, data.y[predict_name])
        df['Pred ' + predict_name] = test_result
        
        print(f"📉 Test Loss (MSE) {predict_name} throughput: {test_loss:.4f}, process time (s):", avg_time)
        
        plot_predicted(df[PredHeaders.time].values, data.y[predict_name], test_result, f'{predict_name}_predicted_{model_type}')
        plot_abs_error_cdf(data.y[predict_name], test_result, f'{predict_name}_abs_error_{model_type}')

    # Save to new CSV file
    df.to_csv(OUTPUT_FOLDER / f'predicted_data_{model_type}.csv', index = False)

if __name__ == "__main__":
    main()