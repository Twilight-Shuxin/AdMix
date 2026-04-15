#!/bin/bash

# nohup ./tmp.sh 2>&1 &
source ./script_utils.sh
source .venv/bin/activate

run_cmd "python run.py --model llm --trace Azure_npref_7b_d2_1 --devices 0,1 --path data/exp --rec-pw --dcgm_group 0" "nohup.out"
run_cmd "python run.py --model llm --trace BurstGPT_npref_7b_d2_1 --devices 0,1 --path data/exp --rec-pw --dcgm_group 0" "nohup.out"
run_cmd "python run.py --model llm,resnet50 --scheduler constant[128] --mps 100,10 --trace BurstGPT_npref_7b_d2_1 --devices 0,1 --path data/exp" "nohup.out"
run_cmd "python run.py --model llm,resnet50 --scheduler constant[128] --mps 100,10 --trace Azure_npref_7b_d2_1 --devices 0,1 --path data/exp" "nohup.out"
run_cmd "python run.py --model llm,resnet50 --scheduler context --trace BurstGPT_npref_7b_d2_1 --devices 0,1 --path data/exp" "nohup.out"
