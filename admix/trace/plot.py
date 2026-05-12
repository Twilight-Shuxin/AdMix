import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import sys

# Load your CSV file
df = pd.read_csv(sys.argv[1])  # <-- change filename
start_time = float(sys.argv[2])
end_time = start_time + 2000

# Filter by time window
df_window = df[(df['Seconds'] >= start_time) & (df['Seconds'] <= end_time)]

# Bin to integer seconds
df_window['SecondBin'] = df_window['Seconds'].apply(lambda x: floor(x))

# Count requests per second
counts = df_window.groupby('SecondBin').size()

# Plot
plt.figure()
plt.bar(counts.index, counts.values)
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Requests")
plt.title(f"Requests from {start_time}s to {end_time}s")
plt.savefig('tmp.png')
