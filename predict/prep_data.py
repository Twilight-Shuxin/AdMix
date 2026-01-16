import os
import sys
import torch
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from predict_models.constants import OUTPUT_FOLDER, SAVED_TEST_DATA_CSV_FILE, PredHeaders, TRAIN_DATA_FILE, VALIDATE_DATA_FILE, TEST_DATA_FILE

# python prep_data.py data.csv
# python prep_data.py data.csv True (all to test data)

def save_processed_data(X, y1, y2, filename):
    X_tensor = torch.tensor(X, dtype = torch.float32)
    y1_tensor = torch.tensor(y1, dtype = torch.float32)
    y2_tensor = torch.tensor(y2, dtype = torch.float32)
    torch.save((X_tensor, y1_tensor, y2_tensor), filename)

def data_prep(data_path, all_test):
    df = pd.read_csv(data_path)
    headers = df.columns.tolist()

    df = df.drop(columns=[PredHeaders.time])

    X = df.drop(columns=[PredHeaders.prefill, PredHeaders.decode])
    y_prefill = df[PredHeaders.prefill]
    y_decode = df[PredHeaders.decode]

    # Split into train + validation (80%) and test (20%) without shuffling
    X_train_all, X_test, y_prefill_train_all, y_prefill_test, y_decode_train_all, y_decode_test = train_test_split(
        X.values, y_prefill.values, y_decode.values, test_size=0.2, random_state=42
    )
    # Split into train (90%) and validation (10%) with shuffling
    X_train, X_validation, y_prefill_train, y_prefill_validation, y_decode_train, y_decode_validation = train_test_split(
        X_train_all, y_prefill_train_all, y_decode_train_all, test_size=0.1, random_state=42
    )

    save_processed_data(X_train, y_prefill_train, y_decode_train, TRAIN_DATA_FILE)
    save_processed_data(X_validation, y_prefill_validation, y_decode_validation, VALIDATE_DATA_FILE)
    print("✅ Data is ready for PyTorch training!")
    print("✅ Data is ready for PyTorch validating!")

    if(all_test):
        X_test = X.values
        y_prefill_test = y_prefill.values
        y_decode_test = y_decode.values

    save_processed_data(X_test, y_prefill_test, y_decode_test, TEST_DATA_FILE)
    print("✅ Data is ready for PyTorch testing! Is all:", all_test)

    df_test = pd.DataFrame(X_test, columns = headers[1:-2])
    df_test.insert(0, PredHeaders.time, [i * 0.1 for i in range(len(y_prefill_test))])
    df_test[PredHeaders.prefill] = y_prefill_test
    df_test[PredHeaders.decode] = y_decode_test

    # Save to CSV
    df_test.to_csv(SAVED_TEST_DATA_CSV_FILE, index=False)


def main():
    data_path = sys.argv[1]
    all_test = False
    if(len(sys.argv) > 2):
        all_test = bool(sys.argv[2])
    data_prep(data_path, all_test)

if __name__ == "__main__":
    main()
