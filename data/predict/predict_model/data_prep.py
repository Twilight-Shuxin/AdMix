import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import sys
from joblib import dump, load

# python data_prep.py train_data.csv train
# python data_prep.py test_data.csv test

def save_processed_data(X, y1, y2, tag):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y1_tensor = torch.tensor(y1, dtype=torch.float32)
    y2_tensor = torch.tensor(y2, dtype=torch.float32)
    torch.save((X_tensor, y1_tensor), 'prefill_' + tag + '.pt')
    torch.save((X_tensor, y2_tensor), 'decode_' + tag + '.pt')

def data_prep(data_path, data_type):
    # 1. Load data
    # df = pd.read_csv('../train_data/llm[BurstGPT].csv')
    df = pd.read_csv(data_path)
    # 2. Drop non-feature columns
    df = df.drop(columns=['Timestamp (s)'])

    # Add square features for non-linear scaling effects
    df["t1 prefill average length squared"] = df["t1 prefill average length"] ** 2
    df["t1 decode average length squared"] = df["t1 decode average length"] ** 2
    df["t1~t2 prefill average length squared"] = df["t1~t2 prefill average length"] ** 2

    # 3. Separate features and targets
    X = df.drop(columns=['Prefill throughput', 'Decode throughput'])
    y_prefill = df['Prefill throughput']
    y_decode = df['Decode throughput']

    # 5. Train-test split
    if(data_type == "train"):
        X_train, X_test, y1_train, y1_test = train_test_split(X.values, y_prefill.values, test_size=0.2, random_state=42)
        _, _, y2_train, y2_test = train_test_split(X.values, y_decode.values, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)
        save_processed_data(X_train_std, y1_train, y2_train, "train")
        save_processed_data(X_test_std, y1_test, y2_test, "validate")
        print("✅ Data is ready for PyTorch training!")
        print("✅ Data is ready for PyTorch validating!")
        dump(sc, 'std_scaler.bin', compress=True)
        print("✅ Data Scaler saved to std_scaler.bin")
    else:
        sc=load('std_scaler.bin')
        X_std = sc.transform(X.values)
        save_processed_data(X_std, y_prefill.values, y_decode.values, "test")
        print("✅ Data is ready for PyTorch testing!")


def main():
    data_path = sys.argv[1]
    data_type = sys.argv[2]
    data_prep(data_path, data_type)

if __name__ == "__main__":
    main()
