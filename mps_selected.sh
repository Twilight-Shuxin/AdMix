#!/bin/bash

# ==============================
# Configuration
# ==============================
trace_id=$1
model=$2
batch_size=$3
shift 3  # Shift away first 3 args, remaining are MPS percentages

# ==============================
# Loop over given MPS values
# ==============================
for mps in "$@"; do
    echo "====================================="
    echo "Testing MPS percentage: $mps"
    echo "====================================="

    # Run command
    cmd="python test_llm.py --model llm,${model} --trace LLM[${trace_id}], --scheduler constant[${batch_size}] --mps ,${mps} --device 1 --profile_group 0 --path data/sighpc/mps"

    echo "$cmd"
    eval "$cmd"

    # Read result
    if [ -f result.log ]; then
        read status passed number < result.log
        echo "Result:" $status $passed $number

        if [ "$status" == "True" ]; then
            echo "✅ Passed at ${mps}%"
        elif [ "$status" == "False" ]; then
            echo "❌ Failed at ${mps}%"
        else
            echo "⚠️ Unknown result in result.log"
        fi
    else
        echo "⚠️ result.log not found!"
    fi

    echo
done
