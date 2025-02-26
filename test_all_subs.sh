#!/bin/bash

# ./test_all_subs.sh $dcgm_group
# models=("resnet50" "vgg19" "inception_v3")
models=("resnet50" "inception_v3")
# model_lens=(18 22 14)
model_lens=(18 14)
strides=(1 2 4)
# batch_sizes=(1 2 4 8 16 32)
batch_sizes=(32)

for i in "${!models[@]}"; do 
    model_len=${model_lens[$i]}
    for batch_size in "${batch_sizes[@]}"; do
        for stride in "${strides[@]}"; do
            echo ./test_subs.sh ${models[$i]} $batch_size $1 $model_len $stride
            ./test_subs.sh ${models[$i]} $batch_size $1 $model_len  $stride
        done
        # python test.py --model ${models[$i]} --batch_size $batch_size --profile_group $1
    done
done
