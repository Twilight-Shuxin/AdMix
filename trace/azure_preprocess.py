import pandas as pd
from datetime import datetime
import sys

# python azure_preprocess.py $raw_filename $output_filename

def main():
    df = pd.read_csv('./raw_data/' + sys.argv[1])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed')
    start_time = df['TIMESTAMP'].iloc[0]
    df['Timestamp'] = (df['TIMESTAMP'] - start_time).dt.total_seconds()
    df = df[['Timestamp', 'ContextTokens', 'GeneratedTokens']]
    df = df.rename(columns={
        'ContextTokens': 'Request tokens',
        'GeneratedTokens': 'Response tokens',
    })
    df.to_csv(sys.argv[2], index=False)
    print("Done. Output saved to " + sys.argv[2])

if __name__ == "__main__":
    main()