#!/bin/bash

for id in {0..5}; do   
    echo python test_llm.py --model vgg19,llm --trace ,BurstGPT_1.csv[$id] --device 1 --test_time 300
    python test_llm.py --model vgg19,llm --trace ,BurstGPT_1.csv[$id] --device 1 --test_time 300
done