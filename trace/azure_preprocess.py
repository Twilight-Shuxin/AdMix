import pandas as pd
from datetime import datetime
import sys

# Load the CSV
df = pd.read_csv('./raw_data/' + sys.argv[1])

# Convert TIMESTAMP column to datetime
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed')

# Get the first timestamp
start_time = df['TIMESTAMP'].iloc[0]

# Compute the time difference in seconds
df['Seconds'] = (df['TIMESTAMP'] - start_time).dt.total_seconds()

# Optional: Move the new column to the front or replace TIMESTAMP
df = df[['Seconds', 'ContextTokens', 'GeneratedTokens']]

# Save the modified CSV
df.to_csv(sys.argv[1].removesuffix(".csv") + '_seconds.csv', index=False)

print("Done. Output saved to " + sys.argv[1].removesuffix(".csv") + '_seconds.csv')
