#!/bin/bash

source ~/MPSTest/script_utils.sh

# Usage: ./eval_data.sh $data_file $output_tag[optional]

run_cmd "python prep_data.py $1 True"

if [ -n "$2" ]; then
    run_cmd "python eval_model.py MLP $2"
else
    run_cmd "python eval_model.py MLP"
fi