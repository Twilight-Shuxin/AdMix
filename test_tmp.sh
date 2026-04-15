#!/bin/bash

# nohup ./test_tmp.sh 2>&1 &
source ./script_utils.sh
source .venv/bin/activate

trace_list=("Azure_npref_7b_d1_1" "BurstGPT_npref_7b_d1_1" "Azure_npref_7b_d1_2" "BurstGPT_npref_7b_d1_2")
trace_list=("Azure_npref_7b_d2_1" "BurstGPT_npref_7b_d2_1")
mps_list=(10 20 10 20)
mps_list=(10 20)
models=("resnet50" "vgg19" "unet3d" "deeplabv3")
declare -A model_mps_bs_map
model_mps_bs_map["resnet50"]=128
model_mps_bs_map["vgg19"]=128
model_mps_bs_map["unet3d"]=16
model_mps_bs_map["deeplabv3"]=32

declare -A model_time_bs_map
model_time_bs_map["resnet50"]=4
model_time_bs_map["vgg19"]=4
model_time_bs_map["unet3d"]=1
model_time_bs_map["deeplabv3"]=2

declare -A llm_model_map
llm_model_map["llama7b"]="meta-llama/Llama-2-7b-hf"
llm_model_map["qwen14b"]="Qwen/Qwen2.5-14B-Instruct"

declare -A llm_devices_map
llm_devices_map["llama7b"]="1"
llm_devices_map["qwen14b"]="0,1"

llm_model_tag="qwen14b"
llm_model="${llm_model_map[$llm_model_tag]}"
devices="${llm_devices_map[$llm_model_tag]}"

gmanager stopmps $devices
trace_list=("BurstGPT_npref_7b_d2_1")
for t_id in "${!trace_list[@]}"; do
    t="${trace_list[$t_id]}"
    run_cmd "python run.py --model llm --llm_model $llm_model --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/$llm_model_tag/utils" "out.out"
done

# ./batch_version.sh 3
gmanager startmps $devices


export ADMIX_LLM_SERVING_PLATFORM=vllm
for t_id in "${!trace_list[@]}"; do
    t="${trace_list[$t_id]}"
    mps="${mps_list[$t_id]}"
    for m in "${models[@]}"; do
        bs="${model_mps_bs_map[$m]}"

        gmanager startmps $devices
        run_cmd "python run.py --model llm,$m --llm_model $llm_model --scheduler constant[$bs] --mps 100,$mps --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/$llm_model_tag/mps" "out_mps.out"
    done
done

# export ADMIX_LLM_SERVING_PLATFORM=vllm
# for t_id in "${!trace_list[@]}"; do
#     t="${trace_list[$t_id]}"
#     for m in "${models[@]}"; do
#         export ADMIX_LLM_MODIFIED=true
#         run_cmd "python run.py --model llm,$m --scheduler context --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data/tp1" "out_context.out"
#     done
# done

# trace_list=("BurstGPT_npref_7b_d1_2")
# gmanager stopmps $devices
# for t_id in "${!trace_list[@]}"; do
#     t="${trace_list[$t_id]}"
#     for m in "${models[@]}"; do
#         t_bs="${model_time_bs_map[$m]}"
#         run_cmd "python run.py --model llm,$m --llm_model $llm_model --scheduler constant[$t_bs] --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/$llm_model_tag/time" "out_time.out"
#     done
# done

# export ADMIX_LLM_SERVING_PLATFORM=vllm
# for t_id in "${!trace_list[@]}"; do
#     t="${trace_list[$t_id]}"
#     mps="${mps_list[$t_id]}"
#     for m in "${models[@]}"; do
#         bs="${model_mps_bs_map[$m]}"
#
#         gmanager startmps $devices
#         run_cmd "python run.py --model llm,$m --llm_model $llm_model --scheduler constant[$bs] --mps 100,$mps --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/$llm_model_tag/mps" "out_mps.out"
#         # run_cmd "python run.py --model llm,$m --llm_model $llm_model --scheduler batch --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data/mod" "out.out"
#
#         # run_cmd "python run.py --model llm,$m --llm_model $llm_model --scheduler context --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/$llm_model_tag/context" "out3.out"
#
#         gmanager stopmps $devices
#         t_bs="${model_time_bs_map[$m]}"
#         run_cmd "python run.py --model llm,$m --llm_model $llm_model --scheduler constant[$t_bs] --dcgm_group 0 --trace $t --devices $devices --rec-pw --path data_staged/$llm_model_tag/time" "out_time.out"
#     done
# done


# tag="vllm_non_prefix_caching_generated"
# run_cmd "python test_vllm_batch.py $tag.csv" "out3.out"
# run_cmd "(cd ./batch_predict/ && ./train_new.sh ./inputs/$tag.csv)" "out3.out"

# tag="vllm_non_prefix_caching_25"
# run_cmd "python test_vllm_batch_trace.py $tag.csv 25"
# run_cmd "(cd ./batch_predict/ && ./eval_data.sh ./inputs/$tag.csv $tag)" "out3.out"

# tag="vllm_non_prefix_caching_26"
# run_cmd "python test_vllm_batch_trace.py $tag.csv 26"
# run_cmd "(cd ./batch_predict/ && ./eval_data.sh ./inputs/$tag.csv $tag)" "out3.out"