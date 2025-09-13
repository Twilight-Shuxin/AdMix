#!/bin/bash

for id in {0..10}; do   
    echo python test_llm.py --model llm --scheduler predictive --trace BurstGPT_1.csv[$id] --device 1 --path data/test_trace
    # python test_llm.py --model llm --scheduler predictive --trace BurstGPT_1.csv[$id] --device 1 --path data/test_trace
done