#!/bin/bash

source ./script_utils.sh

trace_list=(25 26)
mps_list=(20 10)
models=("resnet50" "vgg19" "unet3d" "deeplabv3")
model_batch_sizes=(128 128 16 32)
time_model_batch_sizes=(4 4 1 2)

# export GVLLM_USE_MODIFIED=false
# for t_id in "${!trace_list[@]}"; do
#     t="${trace_list[$t_id]}"
#     run_cmd "python run.py --model llm --dcgm_group 0 --trace $t --device 1 --path data_staged/llama7b/utils" "out.out"
# done

for t_id in "${!trace_list[@]}"; do
    t="${trace_list[$t_id]}"
    mps="${mps_list[$t_id]}"
    for m_id in "${!models[@]}"; do
        m="${models[$m_id]}"
        bs="${model_batch_sizes[$m_id]}"

        export GVLLM_USE_MODIFIED=false
        # run_cmd "python run.py --model llm,$m --scheduler constant[$bs] --mps 100,$mps --dcgm_group 0 --trace $t --device 1 --path data_staged/llama7b/mps" "out.out"
        t_bs="${time_model_batch_sizes[$m_id]}"
        run_cmd "python run.py --model llm,$m --scheduler constant[$t_bs] --dcgm_group 0 --trace $t --device 1 --path data_staged/llama7b/time" "out.out"
        
        # export GVLLM_USE_MODIFIED=true
        # run_cmd "python run.py --model llm,$m --scheduler batch --dcgm_group 0 --trace $t --device 1 --path data_staged/llama7b/batch" "out.out"
    done
done
# tag="vllm_non_prefix_caching_generated"
# run_cmd "python test_vllm_batch.py $tag.csv" "out3.out"
# run_cmd "(cd ./batch_predict/ && ./train_new.sh ./inputs/$tag.csv)" "out3.out"

# tag="vllm_non_prefix_caching_25"
# run_cmd "python test_vllm_batch_trace.py $tag.csv 25"
# run_cmd "(cd ./batch_predict/ && ./eval_data.sh ./inputs/$tag.csv $tag)" "out3.out"

# tag="vllm_non_prefix_caching_26"
# run_cmd "python test_vllm_batch_trace.py $tag.csv 26"
# run_cmd "(cd ./batch_predict/ && ./eval_data.sh ./inputs/$tag.csv $tag)" "out3.out"