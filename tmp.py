import numpy as np
import matplotlib.pyplot as plt
from admix.plotting.monitor_plot import plot_monitored_from_file
from admix.plotting.constants import average_chunks

color_list      = ["#D55E00", "#CC6600", "#0072B2"]  # orange-red, darker orange, blue
line_style_list = ["-", "--", "-"]

from admix.profiling.constants import get_dcgm_data

TAG_CONFIG = {
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
    },
}

def plot_multi_traces(data_prefixes, labels, tag, output_path):
    """
    Plot a single metric (tag) for multiple traces on one figure.

    Args:
        data_prefixes: list of data path prefixes, one per trace
        labels:        list of legend labels matching data_prefixes
        tag:           metric tag, e.g. "sm_active"
        output_path:   path for the saved PDF
    """
    chunk_size = 30
    config = TAG_CONFIG.get(tag, {})
    scale = config.get("scale", 1)

    plt.figure(figsize=(10, 5.5))

    for i, (prefix, label) in enumerate(zip(data_prefixes, labels)):
        _, records = get_dcgm_data(prefix + tag + ".log")
        if len(records) > 20:
            records = records[10:-10]
        else:
            print(f"Warning: {label}: not enough records for trimming.")

        chunked = average_chunks(records, chunk_size)
        values = [v * scale for v in chunked]
        times = [j * chunk_size for j in range(len(values))]

        color = color_list[i % len(color_list)]
        ls = line_style_list[i % len(line_style_list)]
        plt.plot(times, values, linewidth=1.2, label=label, color=color, linestyle=ls, alpha=0.75)

    plt.xlabel("Time (seconds)", fontsize=16, labelpad=12)
    plt.ylabel(config.get("ylabel", tag), fontsize=16)
    if "ylim" in config:
        plt.ylim(*config["ylim"])
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=16, loc="lower left")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    print(f"Saved to {output_path}")


def main():
    tag = "sm_active"
    common_prefix = '/home/k43015/MPSTest/data_new/llama7b/utils/'

    data_prefixes = [
        f"{common_prefix}llm[Azure1,500,100,10800,8,0.01]-d1-",
        f"{common_prefix}llm[Azure3,1,100,10800,8,0.1]-d1-",
        f"{common_prefix}llm[BurstGPT2,200,100,10800,8,1.1]-d1-",
    ]
    labels = ["Azure (2024)", "Azure (2025)", "BurstGPT"]

    plot_multi_traces(data_prefixes, labels, tag, f"{common_prefix}summary/sm_active.pdf")


if __name__ == "__main__":
    main()
