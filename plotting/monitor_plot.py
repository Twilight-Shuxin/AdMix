import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting.constants import average_chunks
from profiling.constants import get_dcgm_data

def plot_monitored(records, data_path_prefix, tag):
    if len(records) > 20:
        records = records[10:-10]
    else:
        print("Warning: plot_monitored: Not enough records for trimming.")

    plt.figure(figsize=(10, 5))
    chunk_size = 15
    records = average_chunks(records, chunk_size)
    df = pd.DataFrame({tag: records})

    tag_config = {
        "sm_active": {
            "ylabel": "SM Activity (%)",
            "scale": 100,
            "ylim": (0, 100),
        },
        "dram_active": {
            "ylabel": "Memory Bandwidth Utilization (%)",
            "scale": 100,
            "ylim": (0, 100),
        },
        "power": {
            "ylabel": "Power (W)",
        }
    }

    config = tag_config.get(tag, {})
    y_label = config.get("ylabel", tag)
    scale = config.get("scale", 1)
    df[tag] = (df[tag] * scale).round(1)

    # Time axis
    df["Time (s)"] = np.arange(len(df)) * chunk_size

    plt.plot(df["Time (s)"], df[tag], linewidth=1.2)
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    if("ylim" in config):
        plt.ylim(*config["ylim"])

    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True)
    plt.tight_layout()

    output_path = f"{data_path_prefix}{tag}.pdf"
    plt.savefig(output_path)

def plot_monitored_from_file(data_prefix, tag):
    _, records = get_dcgm_data(data_prefix + tag + '.log')
    plot_monitored(records, data_prefix, tag)