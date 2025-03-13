#!/bin/bash

# models=("resnet50" "vgg19" "inception_v3")
models=("inception_v3")
batch_sizes=(1 2 4 8 16 32)
percentages=(10 20 30 40 50 60 70 80 90 -1)

for i in "${!models[@]}"; do 
    for batch_size in "${batch_sizes[@]}"; do
        for percentage in "${percentages[@]}"; do
            echo python test.py --model ${models[$i]} --batch_size $batch_size --profile_group $1 --mps $percentage
            python test.py --model ${models[$i]} --batch_size $batch_size --profile_group $1 --mps $percentage
        done
    done
done

./test_mps_comb.sh -1