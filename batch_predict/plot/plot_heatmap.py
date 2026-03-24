import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

input_file = sys.argv[1]
output_file = f'{sys.argv[2]}.png' if len(sys.argv) > 2 else "plot_heatmap.png"
output_file = os.path.join(os.path.dirname(__file__), 'outputs', os.path.basename(output_file))

df = pd.read_csv(input_file)

cached_bins = pd.cut(df["Cached Tkns"], bins=100)
sched_bins = pd.cut(df["Scheduled Tkns"], bins=100)
df["cached_bin"] = cached_bins.apply(lambda x: x.mid).astype(float)
df["sched_bin"] = sched_bins.apply(lambda x: x.mid).astype(float)

has_pred = "Pred Latency" in df.columns
if has_pred:
    df["z_abs"] = (df["Pred Latency"] - df["Latency"]).abs()
    df["z_rel"] = df["z_abs"] / df["Latency"]

    pivot_abs = df.groupby(["cached_bin", "sched_bin"])["z_abs"].mean().unstack()
    pivot_rel = df.groupby(["cached_bin", "sched_bin"])["z_rel"].mean().unstack()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    img1 = ax1.imshow(pivot_abs, aspect="auto", origin="lower", cmap="hot_r")
    plt.colorbar(img1, ax=ax1, label="abs(pred - real)")
    x_ticks = range(0, len(pivot_abs.columns), max(1, len(pivot_abs.columns) // 10))
    y_ticks = range(0, len(pivot_abs.index), max(1, len(pivot_abs.index) // 10))
    ax1.set_xticks(list(x_ticks), [f"{pivot_abs.columns[i]:.0f}" for i in x_ticks], rotation=45)
    ax1.set_yticks(list(y_ticks), [f"{pivot_abs.index[i]:.0f}" for i in y_ticks])
    ax1.set_xlabel("Scheduled Tkns")
    ax1.set_ylabel("Cached Tkns")
    ax1.set_title("Absolute Error Heatmap (mean per cell)")

    img2 = ax2.imshow(pivot_rel, aspect="auto", origin="lower", cmap="hot_r")
    plt.colorbar(img2, ax=ax2, label="abs(pred - real) / real")
    x_ticks = range(0, len(pivot_rel.columns), max(1, len(pivot_rel.columns) // 10))
    y_ticks = range(0, len(pivot_rel.index), max(1, len(pivot_rel.index) // 10))
    ax2.set_xticks(list(x_ticks), [f"{pivot_rel.columns[i]:.0f}" for i in x_ticks], rotation=45)
    ax2.set_yticks(list(y_ticks), [f"{pivot_rel.index[i]:.0f}" for i in y_ticks])
    ax2.set_xlabel("Scheduled Tkns")
    ax2.set_ylabel("Cached Tkns")
    ax2.set_title("Relative Error Heatmap (mean per cell)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
else:
    df["z_val"] = df["Latency"]
    pivot = df.groupby(["cached_bin", "sched_bin"])["z_val"].mean().unstack()

    plt.figure(figsize=(14, 10))
    img = plt.imshow(pivot, aspect="auto", origin="lower", cmap="hot_r")
    plt.colorbar(img, label="Latency (ms)")

    x_ticks = range(0, len(pivot.columns), max(1, len(pivot.columns) // 10))
    y_ticks = range(0, len(pivot.index), max(1, len(pivot.index) // 10))
    plt.xticks(list(x_ticks), [f"{pivot.columns[i]:.0f}" for i in x_ticks], rotation=45)
    plt.yticks(list(y_ticks), [f"{pivot.index[i]:.0f}" for i in y_ticks])

    plt.xlabel("Scheduled Tkns")
    plt.ylabel("Cached Tkns")
    plt.title("Latency Heatmap (mean per cell)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
