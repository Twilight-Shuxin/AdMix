import pandas as pd
import os
import sys

# Specify the directory path 
directory = "../" + sys.argv[1]

# List only .csv filenames
csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
print(csv_files)

# List of CSV files to combine
# csv_files = [
#     'llm[100,11,15000,51]-predict_data.csv', 
#     'llm[100,301,15000,9.5]-predict_data.csv', 
#     'llm[100,5501,15000,1.2]-predict_data.csv',
#     'llm[100,2001,15000,6.1]-predict_data.csv'
#     ]

# Read and concatenate all CSV files
combined_df = pd.concat([pd.read_csv(directory + "/" + file) for file in csv_files], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(sys.argv[1] + '.csv', index=False)

print("✅ All CSVs have been combined into '" + sys.argv[1] + ".csv'")
