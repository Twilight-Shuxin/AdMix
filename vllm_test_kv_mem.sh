#!/bin/bash

# cache_sizes=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
cache_sizes=(0.2 0.4 0.6 0.9)
scales=(200)

for cache_size in "${cache_sizes[@]}"; do
    for scale in "${scales[@]}"; do
        echo export GVLLM_KV_MEM=$cache_size
        export GVLLM_KV_MEM=$cache_size
        echo export GVLLM_TRACE_SCALE=$scale
        export GVLLM_TRACE_SCALE=$scale
        echo python test.py --model llm --profile_group -1 --seqlen 10,10 --trace BurstGPT_1.csv --tag kv$cache_size
        python test.py --model llm --profile_group -1 --seqlen 10,10 --trace BurstGPT_1.csv --tag kv$cache_size
    done
done
