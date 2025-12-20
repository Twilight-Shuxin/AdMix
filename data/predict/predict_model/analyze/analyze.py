import sys

# python analyze.py decode.txt
# python analyze.py prefill.txt 0.1 0.4

def evaluate_predictions(file_path, thresholds=[(0.1, 1), (0.2, 2)]):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            true_val, pred_val = map(float, line.split())
            data.append((true_val, round(pred_val)))

    results = {}
    for t, x in thresholds:
        count_within = sum(
            1 for true, pred in data
            if max(0, min(true * (1 - t), true - x)) <= pred <= max(true * (1 + t), true + x)
        )
        results[t] = count_within

    total = len(data)
    print(f"Total samples: {total}")
    for i, (t, count) in enumerate(results.items()):
        print(f"Threshold {t:.2f}: {count} within range += {thresholds[i]} ({count / total:.2%})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_predictions.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    evaluate_predictions(file_path)