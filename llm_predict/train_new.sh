#!/bin/bash

source ~/MPSTest/script_utils.sh

# Usage: ./train_new.sh $train_data_filename [$model_name]

MODEL=${2:-MLP}

# Remove old model
run_cmd "(cd ./outputs && rm -f model_${MODEL}.* std_scaler.bin)"

# Prepare data, train & eval
run_cmd "python prep_data.py $1"
run_cmd "python train_model.py $MODEL"
run_cmd "python eval_model.py $MODEL"

