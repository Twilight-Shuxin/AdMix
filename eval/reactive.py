import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = "../data_final"
OUTPUT_FOLDER = "./outputs"

MODEL_DIMENSION = {
    "llama7b": "d1",
    "qwen14b": "d2",
    "qwen32b": "d4",
}

TRACE_LISTS = {
    "llama7b": {
        "Azure1,500,100,3600,5,0.022":   "LLaMA-7B - Azure",
        "BurstGPT2,200,100,3600,5,1.5":  "LLaMA-7B - BurstGPT",
    },
    "qwen14b": {
        "Azure1,2000,100,3600,5,0.045":  "Qwen-14B - Azure",
        "BurstGPT2,1000,100,3600,5,1.8": "Qwen-14B - BurstGPT",
    },
    "qwen32b": {
        "Azure1,2000,100,3600,5,0.12":   "Qwen-32B - Azure",
        "BurstGPT1,600,100,3600,5,2.5":  "Qwen-32B - BurstGPT",
    },
}


def load_intensity_log(llm_model, side_model, trace):
    d = MODEL_DIMENSION[llm_model]
    path = os.path.join(
        DATA_ROOT, llm_model, "context",
        f"llm-llm[{trace}]+{side_model}-context-{d}-avg_submitted_intensity.log"
    )
    df = pd.read_csv(path, sep=" ", header=None, names=["time", "intensity"])
    return df


def load_batch_execution_csv(llm_model, side_model, trace):
    d = MODEL_DIMENSION[llm_model]
    path = os.path.join(
        DATA_ROOT, llm_model, "context",
        f"{side_model}-llm[{trace}]+{side_model}-context-{d}-batch_execution.csv"
    )
    df = pd.read_csv(path, usecols=["Time", "MPS"])
    df["MPS"] = pd.to_numeric(df["MPS"], errors="coerce")
    df = df.dropna(subset=["MPS"]).reset_index(drop=True)
    return df


def load_all_csv(llm_model, side_model, trace):
    d = MODEL_DIMENSION[llm_model]
    path = os.path.join(
        DATA_ROOT, llm_model, "context",
        f"{side_model}-llm[{trace}]+{side_model}-context-{d}-all.csv"
    )
    df = pd.read_csv(path, header=None, skiprows=1,
                     names=["device_id", "batch_size", "latency_ms", "mps"])
    return df


def analyze(df):
    df = df[df["mps"] != 100]
    total_latency_ms = df["latency_ms"].sum()

    groups = df.groupby("mps")

    stats = []
    for mps_val, g in groups:
        time_ms = g["latency_ms"].sum()
        time_pct = time_ms / total_latency_ms * 100
        # throughput: items per second for each inference, then average
        throughput = (g["batch_size"] / (g["latency_ms"] / 1000)).mean()
        total_items = g["batch_size"].sum()
        stats.append({
            "mps": mps_val,
            "time_pct": time_pct,
            "avg_throughput": throughput,
            "total_items": total_items,
        })

    df_stats = pd.DataFrame(stats).sort_values("mps").reset_index(drop=True)
    total_items = df_stats["total_items"].sum()
    df_stats["throughput_pct"] = df_stats["total_items"] / total_items * 100
    return df_stats


def plot(stats, batch_exec, intensity, llm_model, side_model, trace):
    tag = TRACE_LISTS.get(llm_model, {}).get(trace, trace)

    mps_labels = [str(int(v)) + "%" for v in stats["mps"]]
    x = np.arange(len(stats))
    bar_width = 0.6

    plt.rcParams.update({"font.size": 15})
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8.5))

    # Top: MPS transitions over time (step plot)
    times = batch_exec["Time"].tolist()
    mps_vals = batch_exec["MPS"].tolist()
    # extend last value to end of timeline for a complete step
    step_times = times + [times[-1]]
    step_mps   = mps_vals + [mps_vals[-1]]
    ax_top.step(step_times, step_mps, where="post", color="#9467bd", linewidth=1.2, label="MPS Percentage")
    ax_top.set_xlabel("(a) Time (s)")
    ax_top.set_ylabel("MPS Percentage (%)")
    ax_top.tick_params(axis="y", labelcolor="#9467bd")
    ax_top.grid(axis="y", linestyle="--", alpha=0.7)
    ax_top.set_axisbelow(True)

    ax_intensity = ax_top.twinx()
    ax_intensity.plot(intensity["time"], intensity["intensity"], color="#DD8452",
                      linewidth=1.2, alpha=0.8, label="LLM Intensity")
    ax_intensity.set_ylabel("Active Requests")
    ax_intensity.tick_params(axis="y", labelcolor="#DD8452")

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_intensity.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, fontsize=15)

    # Bottom: throughput share (bar) + time spent (line)
    ax_bottom.bar(x, stats["throughput_pct"], width=bar_width,
                  color="#6baed6", edgecolor="black", linewidth=0.5, label="Throughput Share")
    ax_bottom.plot(x, stats["time_pct"], "--o", color="#DD8452",
                   linewidth=1.5, markersize=5, label="Time Share")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(mps_labels)
    ax_bottom.set_xlabel("(b) MPS Percentage (%)")
    ax_bottom.set_ylabel("Percentage (%)")
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.7)
    ax_bottom.set_axisbelow(True)
    ax_bottom.legend()


    fig.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, "reactive_vgg19_qwen32b_burstgpt.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def main(llm_model, side_model, trace):
    df = load_all_csv(llm_model, side_model, trace)
    batch_exec = load_batch_execution_csv(llm_model, side_model, trace)
    intensity = load_intensity_log(llm_model, side_model, trace)
    stats = analyze(df)

    print(f"\n{'MPS':>6}  {'Time %':>8}  {'Thpt %':>8}  {'Avg Thpt (items/s)':>20}  {'Total Items':>12}")
    print("-" * 65)
    for _, row in stats.iterrows():
        print(f"{int(row['mps']):>5}%  {row['time_pct']:>7.1f}%  {row['throughput_pct']:>7.1f}%  {row['avg_throughput']:>20.1f}  {int(row['total_items']):>12}")

    plot(stats, batch_exec, intensity, llm_model, side_model, trace)


if __name__ == "__main__":
    # Usage: python reactive.py <llm_model> <side_model> <trace>
    # Example: python reactive.py qwen32b vgg19 "BurstGPT1,600,100,3600,5,2.5"
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Default to the example file
        main("qwen32b", "vgg19", "BurstGPT1,600,100,3600,5,2.5")
