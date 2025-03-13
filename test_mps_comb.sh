#!/bin/bash

models=("resnet50" "vgg19" "inception_v3")
batch_sizes=(1 2 4 8 16 32)
# percentages_i=(40 50 60 70 80 100)
# percentages_j=(80 70 60 50 40 100)
percentages_i=(40 50 60)
percentages_j=(60 50 40)

for i in "${!models[@]}"; do 
    for j in "${!models[@]}"; do
        if [ $i -ge $j ]; then
            continue
        fi
        for batch_size in "${batch_sizes[@]}"; do
            for k in "${!percentages_i[@]}"; do
                echo python test.py --model ${models[$i]},${models[$j]} --batch_size $batch_size,$batch_size --profile_group $1 --mps ${percentages_i[$k]},${percentages_j[$k]}
                python test.py --model ${models[$i]},${models[$j]} --batch_size $batch_size,$batch_size --profile_group $1 --mps ${percentages_i[$k]},${percentages_j[$k]}
            done
        done
    done
done