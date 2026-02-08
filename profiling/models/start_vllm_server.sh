#!/bin/bash 

#meta-llama/Llama-2-7b-chat-hf

python -m vllm.entrypoints.api_server --model "$1"
# export VLLM_LOGGING_LEVEL=ERROR
# python -m vllm.entrypoints.api_server --model "meta-llama/Llama-2-7b-chat-hf"
# python -m vllm.entrypoints.api_server --model "meta-llama/Llama-2-7b-chat-hf" --gpu-memory-utilization 0.7 --disable-log-requests --disable-log-stats
# nohup python -m vllm.entrypoints.api_server --model "meta-llama/Llama-2-7b-chat-hf" --gpu-memory-utilization 0.7 --disable-log-requests --disable-log-stats &

# export CUDA_VISIBLE_DEVICES=0
# curl http://localhost:8000/health

# nohup python -m vllm.entrypoints.api_server --model "openai/gpt-oss-20b" --gpu-memory-utilization 0.7 --disable-log-requests --disable-log-stats &
nohup python -m vllm.entrypoints.api_server --model "meta-llama/Llama-2-7b-chat-hf" --gpu-memory-utilization 0.7 --disable-log-requests --disable-log-stats&
# nohup python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf --gpu-memory-utilization 0.7 --disable-log-requests --disable-log-stats --no-enable-prefix-caching &