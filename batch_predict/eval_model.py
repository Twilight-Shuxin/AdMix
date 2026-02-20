import sys
import pandas as pd
from utils import get_model, load_data
from plot import plot_abs_error_cdf, plot_abs_pct_error_cdf
from predict_models.constants import SAVED_TEST_DATA_CSV_FILE, TEST_DATA_FILE, PredHeaders, OUTPUT_FOLDER

# python eval_model.py MLP $output_tag[optional]
# Output analysis graphs are saved as output_tag + corresponding suffix

def main():
    model_type = sys.argv[1]
    output_tag = sys.argv[2] if len(sys.argv) >= 3 else f'{model_type}'

    df = pd.read_csv(SAVED_TEST_DATA_CSV_FILE)
    data = load_data(TEST_DATA_FILE)

    model = get_model(model_type, data.input_dim)
    model.load()
    test_result, test_loss, avg_time = model.eval(data.X, data.y)
    df['Pred Latency'] = test_result
    
    print(f"📉 Test Loss (MSE) throughput: {test_loss:.4f}, process time (s):", avg_time)

    # Save to new CSV file
    df.to_csv(OUTPUT_FOLDER / f'{output_tag}_pred.csv', index = False)
    plot_abs_error_cdf(df[PredHeaders.latency].values, test_result, f'{output_tag}_abs_error')
    plot_abs_pct_error_cdf(df[PredHeaders.latency].values, test_result, f'{output_tag}_abs_pct_error')


if __name__ == "__main__":
    main()