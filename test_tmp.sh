#!/bin/bash

# nohup ./test_tmp.sh 2>&1 &
source ./script_utils.sh
source .venv/bin/activate

trace_list=("Azure_npref_7b_d1_1" "BurstGPT_npref_7b_d1_1" "Azure_npref_7b_d1_2" "BurstGPT_npref_7b_d1_2")
mps_list=(20 10 20 10)
trace_list=("Azure_npref_7b_d1_1" "BurstGPT_npref_7b_d1_1")
# trace_list=("BurstGPT_npref_7b_d1_2")
# mps_list=(15 10)
# mps_list=(10)
# models=("resnet50" "vgg19" "unet3d" "deeplabv3")
models=("resnet50" "unet3d")
# model_batch_sizes=(128 128 16 32)
model_batch_sizes=(128 16)
time_model_batch_sizes=(4 4 1 2)
devices=1

# gmanager stopmps $devices
# export ADMIX_LLM_MODIFIED=false
# for t_id in "${!trace_list[@]}"; do
#     t="${trace_list[$t_id]}"
#     run_cmd "python run.py --model llm --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/llama7b/utils" "out.out"
# done
./batch_version.sh 3
export ADMIX_LLM_SERVING_PLATFORM=vllm
for t_id in "${!trace_list[@]}"; do
    t="${trace_list[$t_id]}"
    mps="${mps_list[$t_id]}"
    for m_id in "${!models[@]}"; do
        m="${models[$m_id]}"
        bs="${model_batch_sizes[$m_id]}"

        # gmanager startmps $devices
        # export ADMIX_LLM_MODIFIED=false
        # run_cmd "python run.py --model llm,$m --scheduler constant[$bs] --mps 100,$mps --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/llama7b/mps" "out.out"
        export ADMIX_LLM_MODIFIED=true
        run_cmd "python run.py --model llm,$m --scheduler batch --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data/mod" "out.out"
        # gmanager stopmps $devices
        # t_bs="${time_model_batch_sizes[$m_id]}"
        # run_cmd "python run.py --model llm,$m --scheduler constant[$t_bs] --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/llama7b/time" "out.out"
    done
done
# cp "$new_file" "$dest_file"
# tag="vllm_non_prefix_caching_generated"
# run_cmd "python test_vllm_batch.py $tag.csv" "out3.out"
# run_cmd "(cd ./batch_predict/ && ./train_new.sh ./inputs/$tag.csv)" "out3.out"

# tag="vllm_non_prefix_caching_25"
# run_cmd "python test_vllm_batch_trace.py $tag.csv 25"
# run_cmd "(cd ./batch_predict/ && ./eval_data.sh ./inputs/$tag.csv $tag)" "out3.out"

# tag="vllm_non_prefix_caching_26"
# run_cmd "python test_vllm_batch_trace.py $tag.csv 26"
# run_cmd "(cd ./batch_predict/ && ./eval_data.sh ./inputs/$tag.csv $tag)" "out3.out"