import pandas as pd
import sys

# === Input CSV ===
csv_file = sys.argv[1]   # change this to your actual file name

# === Read CSV ===
df = pd.read_csv(csv_file)

# === Extract relevant columns ===
prefill_cols = ["Prefill throughput", "Predicted prefill"]
decode_cols = ["Decode throughput", "Predicted decode"]

# === Output TXT files ===
with open("prefill.txt", "w") as f:
    for i, row in df.iterrows():
        f.write(f"{row[prefill_cols[0]]} {row[prefill_cols[1]]}\n")

with open("decode.txt", "w") as f:
    for i, row in df.iterrows():
        f.write(f"{row[decode_cols[0]]} {row[decode_cols[1]]}\n")

print("✅ Files generated: prefill.txt and decode.txt")
