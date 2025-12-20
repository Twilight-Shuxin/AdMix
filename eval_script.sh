#!/bin/bash

# Define lists
numbers=(16 11)  
# numbers=(19) 
numbers=(19 21)
models=("resnet50" "vgg19" "unet3d" "deeplabv3") # change to your model names
# models=("vgg19")
model_batch_sizes=(128 128 16 32)
# model_batch_sizes=(128)

cmd="python test_llm.py --model llm --trace LLM[23] --scheduler predictive --profile_group 0 --device 1 --path data/util > eval_trace.out"
echo "$cmd"
eval "$cmd"

cmd="python test_llm.py --model llm --trace LLM[22] --scheduler predictive --profile_group 0 --device 1 --path data/util >> eval_trace.out"
echo "$cmd"
eval "$cmd"

exit

# Loop over each number
for x in "${numbers[@]}"; do
  # Loop over each model
  for i in "${!models[@]}"; do
    model_name="${models[$i]}"
    batch_size="${model_batch_sizes[$i]}"
    # Command 1: AIMD scheduler
    # cmd1="python test_llm.py --model llm,${model_name} --trace LLM[${x}], --scheduler AIMD --profile_group 0 --path data/sighpc/aimd --device 1 >> out_eval.out"
    # echo "$cmd1"
    # eval "$cmd1"
    # # Command 2: predictive_v2 scheduler
    cmd4="./mps_selected.sh ${x} ${model_name} ${batch_size} 20 >> out_eval_mps.out"
    echo "$cmd4"
    eval "$cmd4"

    # cmd2="python test_llm.py --model llm,${model_name} --trace LLM[${x}], --scheduler predictive_v2 --profile_group 0 --path data/sighpc/predictive_v2 --device 1 >> out_eval.out"
    # echo "$cmd2"
    # eval "$cmd2"

    # cmd3="./binary_mps.sh ${x} ${model_name} ${batch_size} >> out_mps.out"
    # echo "$cmd3"
    # eval "$cmd3"
  done
done


# cmd="python test_llm.py --model llm --trace LLM[15] --profile_group 0 --scheduler predictive --path data/util --device 1 >> out_tmp2.out"
# echo "$cmd"
# eval "$cmd"