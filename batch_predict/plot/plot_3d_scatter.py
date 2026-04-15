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
    plt.colorbar(sc, ax=ax, label="Latency (ms)")
    ax.set_xlabel("Cached Tkns")
    ax.set_ylabel("Scheduled Tkns")
    ax.set_title("3D Scatter: Latency vs Token Dims")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
