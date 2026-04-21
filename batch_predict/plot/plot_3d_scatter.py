import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_file = sys.argv[1]
output_file = f'{sys.argv[2]}.pdf' if len(sys.argv) > 2 else "plot_3d_scatter.pdf"
output_file = os.path.join(os.path.dirname(__file__), 'outputs', os.path.basename(output_file))

df = pd.read_csv(input_file)

has_pred = "Pred Latency" in df.columns
if has_pred:
    z_abs = (df["Pred Latency"] - df["Latency"]).abs()
    z_rel = z_abs / df["Latency"]

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(df["Cached Tkns"], df["Scheduled Tkns"], z_abs, c=z_abs, cmap="viridis", s=0.5, alpha=0.3, rasterized=True)
    plt.colorbar(sc1, ax=ax1, label="abs(pred - real)")
    ax1.set_xlabel("Cached Tkns")
    ax1.set_ylabel("Scheduled Tkns")
    ax1.set_title("3D Scatter: Absolute Error")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(df["Cached Tkns"], df["Scheduled Tkns"], z_rel, c=z_rel, cmap="viridis", s=0.5, alpha=0.3, rasterized=True)
    plt.colorbar(sc2, ax=ax2, label="abs(pred - real) / real")
    ax2.set_xlabel("Cached Tkns")
    ax2.set_ylabel("Scheduled Tkns")
    ax2.set_title("3D Scatter: Relative Error")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
else:
    z = df["Latency"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(z.max())
    sc = ax.scatter(df["Cached Tkns"], df["Scheduled Tkns"], z, c=z, cmap="viridis", s=0.5, alpha=0.3, rasterized=True)
    fs = 8
    cbar = plt.colorbar(sc, ax=ax, label="Latency (ms)", shrink=0.5, pad=0.03)
    cbar.set_label("Latency (ms)", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)

    # Skip the tick at 0: in 3D, the label at the origin corner renders
    # displaced far from the axis, so start ticks from the first non-zero step.
    x_max = df["Cached Tkns"].max()
    x_ticks = range(20000, int(x_max) + 1, 20000)
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels([f"{x // 1000}k" for x in x_ticks], fontsize=fs)

    y_max = df["Scheduled Tkns"].max()
    y_ticks = range(500, int(y_max) + 1, 500)
    ax.set_yticks(list(y_ticks))
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=fs)
    ax.set_xlabel("Cached Tkns", fontsize=fs, labelpad=-8)
    ax.set_ylabel("Scheduled Tkns", fontsize=fs, labelpad=-4)
    ax.tick_params(axis='x', pad=-4)
    ax.tick_params(axis='y', pad=-2)
    ax.tick_params(axis='z', pad=-1, labelsize=fs)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
