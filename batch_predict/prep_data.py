import os
import sys
import torch
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from predict_models.constants import OUTPUT_FOLDER, SAVED_TEST_DATA_CSV_FILE, PredHeaders, TRAIN_DATA_FILE, VALIDATE_DATA_FILE, TEST_DATA_FILE

# python prep_data.py data.csv
# python prep_data.py data.csv True (all to test data)

def save_processed_data(X, y, filename):
    X_tensor = torch.tensor(X, dtype = torch.float32)
    y_tensor = torch.tensor(y, dtype = torch.float32)
    torch.save((X_tensor, y_tensor), filename)

def data_prep(data_path, all_test):
    headers = pd.read_csv(data_path, nrows=0)
    n_cols = len(headers.columns)
    df = pd.read_csv(
        data_path,
        usecols=range(n_cols),
        engine="python"
    )

    X = df.drop(columns=[PredHeaders.latency])
    y = df[PredHeaders.latency]

    # Split into train + validation (80%) and test (20%) without shuffling
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )
    # Split into train (90%) and validation (10%) with shuffling
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_all, y_train_all, test_size=0.1, random_state=42
    )

    if(all_test):
        X_test = X.values
        y_test = y.values
    else:
        save_processed_data(X_train, y_train, TRAIN_DATA_FILE)
        save_processed_data(X_validation, y_validation, VALIDATE_DATA_FILE)
        print("✅ Data is ready for PyTorch training!")
        print("✅ Data is ready for PyTorch validating!")

    save_processed_data(X_test, y_test, TEST_DATA_FILE)
    print("✅ Data is ready for PyTorch testing! Is all:", all_test)

    df_test = pd.DataFrame(X_test, columns = X.columns.to_list())
    df_test[PredHeaders.latency] = y_test

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
