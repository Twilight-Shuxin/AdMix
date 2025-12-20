import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import sys
from joblib import dump, load
import os
from plot import plot_distributions

# python data_prep.py data.csv

def save_processed_data(X, y1, y2, tag):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y1_tensor = torch.tensor(y1, dtype=torch.float32)
    y2_tensor = torch.tensor(y2, dtype=torch.float32)
    torch.save((X_tensor, y1_tensor), 'prefill_' + tag + '.pt')
    torch.save((X_tensor, y2_tensor), 'decode_' + tag + '.pt')

def data_prep(data_path, all_test):
    # 1. Load data
    # df = pd.read_csv('../train_data/llm[BurstGPT].csv')
    df = pd.read_csv(data_path)

    # Add square features for non-linear scaling effects
    df.insert(len(df.columns) - 2 , "t1 prefill average length squared", df["t1 prefill average length"] ** 2)
    df.insert(len(df.columns) - 2 , "t1 decode average length squared", df["t1 decode average length"] ** 2)
    df.insert(len(df.columns) - 2 , "t1~t2 prefill average length squared", df["t1~t2 prefill average length"] ** 2)
    df = df.drop(columns=['t1~t2 finished', 't1~t2 finished average throughput'])
    headers = df.columns.tolist()

    # 2. Drop non-feature columns
    df = df.drop(columns=['Timestamp (s)'])

    # 3. Separate features and targets
    X = df.drop(columns=['Prefill throughput', 'Decode throughput'])
    y_prefill = df['Prefill throughput']
    y_decode = df['Decode throughput']

    # Split into train + validation (80%) and test (20%) without shuffling
    X_train_all, X_test, y_prefill_train_all, y_prefill_test, y_decode_train_all, y_decode_test = train_test_split(
        X.values, y_prefill.values, y_decode.values, test_size=0.2, random_state=42
    )
    # Split into train (90%) and validation (10%) with shuffling
    X_train, X_validation, y_prefill_train, y_prefill_validation, y_decode_train, y_decode_validation = train_test_split(
        X_train_all, y_prefill_train_all, y_decode_train_all, test_size=0.1, random_state=42
    )
    sc = StandardScaler()
    model_path = os.getenv("MODEL_PATH", "")
    plot_distributions(X_train, y_prefill_train, y_decode_train, headers[1:], 'train')
    if(len(model_path) != 0):
        model_path += "/"
        sc=load(model_path + 'std_scaler.bin')
        X_train_std = sc.transform(X_train)
    else:
        X_train_std = sc.fit_transform(X_train)
    X_validation_std = sc.transform(X_validation)

    save_processed_data(X_train_std, y_prefill_train, y_decode_train, "train")
    save_processed_data(X_validation_std, y_prefill_validation, y_decode_validation, "validate")
    print("✅ Data is ready for PyTorch training!")
    print("✅ Data is ready for PyTorch validating!")
    if(len(model_path) == 0):
        dump(sc, 'std_scaler.bin', compress=True)
        print("✅ Data Scaler saved to std_scaler.bin")

    if(all_test):
        X_test = X.values
        y_prefill_test = y_prefill.values
        y_decode_test = y_decode.values
    plot_distributions(X_test, y_prefill_test, y_decode_test, headers[1:], 'test')
    X_test_std = sc.transform(X_test)
    save_processed_data(X_test_std, y_prefill_test, y_decode_test, "test")
    print("✅ Data is ready for PyTorch testing! All:", all_test)

    # Combine test data back into one DataFrame
    df_test = pd.DataFrame(X_test, columns=headers[1:-2])  # X_test retains feature names
    df_test.insert(0, headers[0], [i * 0.1 for i in range(len(y_prefill_test))])
    df_test[headers[-2]] = y_prefill_test
    df_test[headers[-1]] = y_decode_test

    # Save to CSV
    df_test.to_csv("test_data_saved.csv", index=False)


def main():
    data_path = sys.argv[1]
    all_test = False
    if(len(sys.argv) > 2):
        all_test = bool(sys.argv[2])
    data_prep(data_path, all_test)

if __name__ == "__main__":
    main()
