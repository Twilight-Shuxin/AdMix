#!/bin/bash

# ==============================
# Configuration
# ==============================
model=$2
batch_size=$3

low=10
high=100
best=0
step=10

# Round a number to the nearest multiple of step
round_to_step() {
    local val=$1
    echo $(( (val + step / 2) / step * step ))
}

mid=20

# ==============================
# Binary Search Loop
# ==============================
while [ $low -le $high ]; do
    # Ensure within bounds
    if [ $mid -gt $high ]; then
        mid=$high
    fi
    if [ $mid -lt $low ]; then
        mid=$low
    fi

    echo "Testing percentage: $mid"

    # Run test
    cmd="python test_llm.py --model llm,"$model" --trace LLM[$1], --scheduler constant["$batch_size"] --mps ,"$mid" --device 1 --profile_group 0 --path data/bmps"
    echo "$cmd"
    eval "$cmd"
    read status passed number < result.log
    echo "Binary search result:" $status $passed $number
    # Check result.log for true/false
    if [ "$status" == "True" ]; then
        echo "✅ Passed at $mid%"
        best=$mid
        low=$((mid + step))
    elif [ "$status" == "False" ]; then
        echo "❌ Failed at $mid%"
        high=$((mid - step))
    fi
    mid=$(( (low + high) / 2 ))
    mid=$(round_to_step $mid)
done