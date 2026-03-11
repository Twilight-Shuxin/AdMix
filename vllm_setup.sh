#!/bin/bash

# File definitions
VLLM_FOLDER=".venv/lib/python3.12/site-packages/vllm/v1/engine"
original_file=${VLLM_FOLDER}/core_original.py
modified_file=${VLLM_FOLDER}/core_shuxin.py
test_file=${VLLM_FOLDER}/core_test.py
vllm_file=${VLLM_FOLDER}/core.py

# Check argument
if [ $# -eq 0 ]; then
    # No argument → use original
    echo "Use original version:" $original_file
    cp "$original_file" "$vllm_file"
elif [ "$1" = "modified" ]; then
    # Argument is "modified" → use modified
    echo "Use modified version:" $modified_file
    cp "$modified_file" "$vllm_file"
elif [ "$1" = "test" ]; then
    echo "Use version for testing:" $test_file
    cp "$test_file" "$vllm_file"
else
    echo "Usage: $0 [modified / test]"
    exit 1
fi
