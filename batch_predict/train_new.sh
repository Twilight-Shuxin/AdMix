#!/bin/bash

source ~/MPSTest/script_utils.sh

# Usage: ./train_new.sh $train_data_filename

# Remove old model
run_cmd "(cd ./outputs && rm -f model_MLP.pth std_scaler.bin)"

# Prepare data, train & eval
run_cmd "python prep_data.py $1"
run_cmd "python train_model.py MLP"
run_cmd "python eval_model.py MLP"

