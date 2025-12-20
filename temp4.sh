#!/bin/bash

for mps in $(seq 10 10 100); do
    # for bs in $(seq 8 16 128); do
    # for bs in $(seq 1 7) $(seq 8 16 128); do
    for bs in $(seq 9 23); do
        echo "Running batch size $bs"
        cmd="python test_llm.py --model resnet50 --profile_group 0 --mps $mps --device 1 --batch_size $bs --path data/tmp --test_time 15"
        echo "$cmd"
        eval "$cmd"
    done
done
