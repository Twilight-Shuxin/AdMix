#!/bin/bash

# File definitions
VLLM_FOLDER=".venv/lib/python3.12/site-packages/vllm/v1/engine"
SGLANG_FOLDER="./plugins/sglang_env/.venv/lib/python3.12/site-packages/sglang/srt/managers"
VLLM_PLUGIN_FOLDER="./plugins/vllm/engine"
SGLANG_PLUGIN_FOLDER="./plugins/sglang"

# Select platform based on ADMIX_LLM_SERVING_PLATFORM
if [ "$ADMIX_LLM_SERVING_PLATFORM" = "sglang" ]; then
    original_file=${SGLANG_PLUGIN_FOLDER}/scheduler.py
    modified_file=${SGLANG_PLUGIN_FOLDER}/scheduler_shuxin.py
    dest_file=${SGLANG_FOLDER}/scheduler.py
else
    original_file=${VLLM_PLUGIN_FOLDER}/core_original.py
    modified_file=${VLLM_PLUGIN_FOLDER}/core_shuxin.py
    test_file=${VLLM_PLUGIN_FOLDER}/core_test.py
    dest_file=${VLLM_FOLDER}/core.py
fi

echo "Platform: ${ADMIX_LLM_SERVING_PLATFORM:-vllm}"

# Check argument
if [ $# -eq 0 ]; then
    # No argument → use original
    echo "Use original version:" $original_file
    cp "$original_file" "$dest_file"
elif [ "$1" = "modified" ]; then
    # Argument is "modified" → use modified
    echo "Use modified version:" $modified_file
    cp "$modified_file" "$dest_file"
elif [ "$1" = "test" ]; then
    if [ "$ADMIX_LLM_SERVING_PLATFORM" = "sglang" ]; then
        echo "No test version for sglang, using original"
        cp "$original_file" "$dest_file"
    else
        echo "Use version for testing:" $test_file
        cp "$test_file" "$dest_file"
    fi
else
    echo "Usage: $0 [modified / test]"
    exit 1
fi
