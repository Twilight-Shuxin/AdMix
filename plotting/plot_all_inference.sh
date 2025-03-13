#!/bin/bash

models=("resnet50" "vgg19" "inception_v3")
batch_sizes=(1 2 4 8 16 32)
percentages_i=(40 50 60 70 80 100)
percentages_j=(80 70 60 50 40 100)

for i in "${!models[@]}"; do 
    for j in "${!models[@]}"; do
        if [ $i -ge $j ]; then
            continue
        fi
        for batch_size in "${batch_sizes[@]}"; do
            python plot_interference.py data data_mps ${models[$i]},${models[$j]} $batch_size,$batch_size
            echo python plot_interference.py data data_mps ${models[$i]},${models[$j]} $batch_size,$batch_size
        done
    done
done