import pandas as pd
import os
import sys
import glob

# Step 1: Load all CSV file paths
csv_files = glob.glob("*.csv")

# Step 2: Initialize empty list and cumulative offset
dfs = []
timestamp_offset = 0

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    # Adjust timestamps if not the first file
    if i > 0:
        df["Timestamp (s)"] += timestamp_offset + 5  # Add 5s gap from last timestamp
    # Append df to list
    dfs.append(df)
    # Update offset for next iteration
    timestamp_offset = df["Timestamp (s)"].max()

# Step 3: Concatenate all DataFrames
final_df = pd.concat(dfs, ignore_index=True)

# Optionally save to CSV
final_df.to_csv("llm[BurstGPT].csv", index=False)

