import sys
import csv

def main():
    csv_path = sys.argv[1]
    latency = []
    pred_latency = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # convert to float (strip just in case)
            l = float(row["Latency"].strip())
            pl = float(row["Pred Latency"].strip())

            latency.append(l)
            pred_latency.append(pl)

    if not latency:
        raise ValueError("CSV contains no data rows")
    mse = sum((pl - l) ** 2 for pl, l in zip(pred_latency, latency)) / len(latency)

    print(f"MSE (Pred Latency vs Latency): {round(mse,3)}")

if __name__ == "__main__":
    main()