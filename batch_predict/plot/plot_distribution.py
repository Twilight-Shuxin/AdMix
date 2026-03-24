import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = sys.argv[1]
output_file = f'{sys.argv[2]}.png' if len(sys.argv) > 2 else "plot_distribution.png"
output_file = os.path.join(os.path.dirname(__file__), 'outputs', os.path.basename(output_file))

df = pd.read_csv(input_file)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

SCHED_BIN_SIZE = 200
CACHED_BIN_SIZE = 1000

sched_edges = np.arange(df["Scheduled Tkns"].min(),
                        df["Scheduled Tkns"].max() + SCHED_BIN_SIZE,
                        SCHED_BIN_SIZE)
cached_edges = np.arange(df["Cached Tkns"].min(),
                         df["Cached Tkns"].max() + CACHED_BIN_SIZE,
                         CACHED_BIN_SIZE)

# Figure 1: distribution of dots by Scheduled Tkns
sched_counts, _ = np.histogram(df["Scheduled Tkns"], bins=sched_edges)
sched_centers = (sched_edges[:-1] + sched_edges[1:]) / 2
ax1.bar(sched_centers, sched_counts, width=SCHED_BIN_SIZE * 0.9)
ax1.set_xlabel("Scheduled Tkns")
ax1.set_ylabel("Count")
ax1.set_title(f"Distribution by Scheduled Tkns (bin size={SCHED_BIN_SIZE})")

# Figure 2: distribution of dots by Cached Tkns
cached_counts, _ = np.histogram(df["Cached Tkns"], bins=cached_edges)
cached_centers = (cached_edges[:-1] + cached_edges[1:]) / 2
ax2.bar(cached_centers, cached_counts, width=CACHED_BIN_SIZE * 0.9)
ax2.set_xlabel("Cached Tkns")
ax2.set_ylabel("Count")
ax2.set_title(f"Distribution by Cached Tkns (bin size={CACHED_BIN_SIZE})")

# Figure 3: 2D distribution (heatmap of counts)
h, xedges, yedges = np.histogram2d(df["Scheduled Tkns"], df["Cached Tkns"],
                                    bins=[sched_edges, cached_edges])
img = ax3.imshow(h.T, origin="lower", aspect="auto",
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                 cmap="YlOrRd")
plt.colorbar(img, ax=ax3, label="Count")
ax3.set_xlabel("Scheduled Tkns")
ax3.set_ylabel("Cached Tkns")
ax3.set_title("2D Distribution (Scheduled vs Cached Tkns)")

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Saved to {output_file}")
