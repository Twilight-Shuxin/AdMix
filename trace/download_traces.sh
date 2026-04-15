#!/bin/bash

source ../.venv/bin/activate
wget -O BurstGPT1.csv https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv
wget -O BurstGPT2.csv https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_2.csv
mkdir -p ./raw_data && wget -P ./raw_data/ https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv
wget -P ./raw_data/ https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv

python azure_preprocess.py AzureLLMInferenceTrace_conv_1week.csv Azure1.csv
python azure_preprocess.py AzureLLMInferenceTrace_conv.csv Azure2.csv