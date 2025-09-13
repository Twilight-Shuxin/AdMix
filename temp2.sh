#!/bin/bash

# for id in {0..5}; do   
#     echo python test_llm.py --model vgg19,llm --trace ,BurstGPT_1.csv[$id] --device 1 --test_time 300
#     python test_llm.py --model vgg19,llm --trace ,BurstGPT_1.csv[$id] --device 1 --test_time 300
# done

python test_llm.py --model llm --scheduler predictive --trace BurstGPT_1.csv[12] --device 1 --path data/test_trace