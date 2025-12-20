import pandas as pd
import matplotlib.pyplot as plt
import sys

def average_chunks(data, chunk_size):
    """Return a list where each item is the average of every `chunk_size` items."""
    return [
        sum(data[i:i + chunk_size]) / len(data[i:i + chunk_size])
        for i in range(0, len(data), chunk_size)
    ]

def plot_dcgm(records, data_path_prefix, tag):
    plt.figure(figsize=(10, 5))
    records = records[10:-10]
    chunk_size = 30
    records = average_chunks(records, chunk_size)
    df = pd.DataFrame(records, columns=[tag])
    data = df[tag].copy()
    y_tag = tag
    if(tag == "sm_active"):
        y_tag = "SM Activity (%)"
        data = [round(x * 100, 1) for x in data]
        plt.ylim(0, 100)
    elif(tag == "dram_active"):
        y_tag = "Memory Bandwidth Utilization (%)"
        data = [round(x * 100, 1) for x in data]
        plt.ylim(0, 100)
    # Add a time column (1 second per row)
    df["Time (s)"] = [i * chunk_size for i in range(len(records))]
    plt.plot(df["Time (s)"], data, linewidth=1.2)
    # plt.title(tag + " Over Time")
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel(y_tag, fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(data_path_prefix + tag + '.pdf')

def plot_dcgm_from_file(data_prefix, tag):
    file_path = data_prefix + tag + '.log'
    # Read the file (skip lines starting with '#')
    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line == "ID":
                continue
            parts = line.split()
            records.append(float(parts[-1]))
    if not records:
        raise ValueError("No valid data found in file.")
    plot_dcgm(records, data_prefix, tag)

def main():
    plot_dcgm_from_file(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
