#!/bin/bash

# ./test_subs.sh resnet50 4 -1 18 9
export GVLLM_PARTIAL_STRIDE=$5
for ((i=0; i<$4; i+=$5)); do
    rr=$(($i+$5))
    rr=$(($rr<=$4 ? $rr : $4))
    echo python test_llm.py --model $1 --batch_size $2 --profile_group $3 --device 1 --start $i --end $rr --test_time 40
    python test_llm.py --model $1 --batch_size $2 --profile_group $3 --device 1 --start $i --end $rr --test_time 40
    # python test.py --model $1 --batch_size $2 --profile_group $3 --start $i --end $(($i + 1))
done