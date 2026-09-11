#!/bin/bash

source ~/AdMix/script_utils.sh

# Usage: ./eval_data.sh $data_file $output_tag [$model_name]

MODEL=${3:-MLP}

run_cmd "python prep_data.py $1 True"
run_cmd "python eval_model.py $MODEL $2"
run_cmd "python ./plot/plot_3d_scatter.py ./outputs/eval/${2}_pred.csv ${2}_scatter"