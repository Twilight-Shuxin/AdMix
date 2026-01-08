import pandas as pd
import matplotlib.pyplot as plt
import sys
import ones

from plotting.utils import average_chunks

def plot_throughput(records, file_path):
    chunk_size = 60
    records = average_chunks(records, chunk_size)
    print(len(records))
    df = pd.DataFrame(records, columns=["Time", "Throughput"])
    # Add a time column (1 second per row)
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["Throughput"], linewidth=1.2)
    # plt.title(tag + " Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("DNN Workload")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.splitext(file_path)[0] + ".pdf")

def plot_throughput_from_file(file_path):
    # Read the file (skip lines starting with '#')
    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            records.append((float(parts[0]), float(parts[1])))
    if not records:
        raise ValueError("No valid data found in file.")
    plot_throughput(records, file_path)

def main():
    plot_throughput_from_file(sys.argv[1])

if __name__ == "__main__":
    main()
