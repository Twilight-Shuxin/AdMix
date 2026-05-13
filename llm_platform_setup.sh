#!/bin/bash

# File definitions
VLLM_FOLDER=".venv/lib/python3.12/site-packages/vllm/v1/engine"
SGLANG_FOLDER="./engines/sglang_env/.venv/lib/python3.12/site-packages/sglang/srt/managers"
VLLM_MODIFIED_FOLDER="./engines/vllm/engine"
SGLANG_MODIFIED_FOLDER="./engines/sglang"

# Select platform based on ADMIX_LLM_SERVING_PLATFORM
if [ "$ADMIX_LLM_SERVING_PLATFORM" = "sglang" ]; then
    original_file=${SGLANG_MODIFIED_FOLDER}/scheduler.py
    modified_file=${SGLANG_MODIFIED_FOLDER}/scheduler_modified.py
    dest_file=${SGLANG_FOLDER}/scheduler.py
else
    original_file=${VLLM_MODIFIED_FOLDER}/core.py
    modified_file=${VLLM_MODIFIED_FOLDER}/core_modified.py
    dest_file=${VLLM_FOLDER}/core.py
fi

echo "Platform: ${ADMIX_LLM_SERVING_PLATFORM:-vllm}"

# Check argument
if [ $# -eq 0 ]; then
    # No argument → use original
    echo "LLM platform: use original version @ " $original_file
    cp "$original_file" "$dest_file"
elif [ "$1" = "modified" ]; then
    # Argument is "modified" → use modified
    echo "LLM platform: use modified version @ " $modified_file
    cp "$modified_file" "$dest_file"
else
    echo "Usage: $0 [modified]"
    exit 1
fi
