import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load the CSV
df = pd.read_csv(sys.argv[1])

df_prefill = df[df["Prefill throughput"] != 0]
df_decode = df[df["Decode throughput"] != 0]

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# --- Prefill throughput ---
counts, bins, patches = axes[0].hist(
    df_prefill["Prefill throughput"],
    bins=20,
    color='skyblue',
    edgecolor='black',
    alpha=0.7
)
axes[0].set_title("Prefill Throughput Distribution")
axes[0].set_xlabel("Throughput")
axes[0].set_ylabel("Count")
axes[0].grid(alpha=0.3)

# Add labels above each bar
for count, patch in zip(counts, patches):
    if count > 0:  # skip empty bars
        axes[0].text(
            patch.get_x() + patch.get_width() / 2,
            count,
            f"{int(count)}",
            ha='center',
            va='bottom',
            fontsize=8
        )

# --- Decode throughput ---
counts, bins, patches = axes[1].hist(
    df_decode["Decode throughput"],
    bins=20,
    color='salmon',
    edgecolor='black',
    alpha=0.7
)
axes[1].set_title("Decode Throughput Distribution")
axes[1].set_xlabel("Throughput")
axes[1].set_ylabel("Count")
axes[1].grid(alpha=0.3)

# Add labels
for count, patch in zip(counts, patches):
    if count > 0:
        axes[1].text(
            patch.get_x() + patch.get_width() / 2,
            count,
            f"{int(count)}",
            ha='center',
            va='bottom',
            fontsize=8
        )

plt.tight_layout()
plt.show()
plt.savefig('distribution.png')