#!/bin/bash 

#meta-llama/Llama-2-7b-chat-hf

python -m vllm.entrypoints.api_server --model "$1"
# python -m vllm.entrypoints.api_server --model "meta-llama/Llama-2-7b-chat-hf"
