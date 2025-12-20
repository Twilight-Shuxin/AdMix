#!/bin/bash

id=$1
ll=$2
rr=$3
step=0.5   # in 0.5 multiples: step=1 means 0.5 after scaling

if [ -z "$id" ] || [ -z "$ll" ] || [ -z "$rr" ]; then
    echo "Usage: $0 <id> <low> <high> [step]"
    exit 1
fi

# Scale everything by 2
low=$(awk "BEGIN {printf \"%d\", $ll*2}")
high=$(awk "BEGIN {printf \"%d\", $rr*2}")
step=$(awk "BEGIN {printf \"%d\", $step*2}")
best=0
best_number=0

while [ $low -le $high ]; do
    mid=$(( (low + high) / 2 ))
    scale=$(awk "BEGIN {printf \"%.1f\", $mid/2}") # convert back to float for logging

    echo "Testing scale: $scale"

    # Update config and run test
    python update_trace_config.py $id --update scale=$scale
    python test_llm.py --model llm --trace "LLM[$id]" --device 1

    read status passed submitted < result.log

    if [ "$status" == "True" ]; then
        echo "✅ Passed at scale=$scale, passed=$passed submitted=$submitted"

        # Update last_true and last_true_submitted
        python update_trace_config.py $id --update last_true=$scale last_true_submitted=$submitted

        if [ $((submitted - passed)) -lt 10 ]; then
            echo "✅ Strictly passed at scale=$scale"
            python update_trace_config.py $id --update last_all_passed=$scale
            best=$scale
            low=$((mid + step))   # try higher
        else
            echo "❌ Not strictly passed at scale=$scale"
            high=$((mid - step))  # try lower
        fi
    else
        echo "❌ Failed at scale=$scale, passed=$passed, submitted=$submitted"
        python update_trace_config.py $id --update last_false=$scale
        high=$((mid - step))
    fi
    echo $low $high
done

echo
echo "=============================="
echo "✅ Maximum passing scale: $best"
echo "=============================="
