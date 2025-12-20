#!/bin/bash

models=("decision_tree" "random_forest" "linear" "poisson" "MLP")

for i in "${!models[@]}"; do 
    echo python train_model.py ${models[$i]} prefill &
    python train_model.py ${models[$i]} prefill &
    pid1=$!
    echo python train_model.py ${models[$i]} decode &
    python train_model.py ${models[$i]} decode &
    pid2=$!
    wait $pid1
    wait $pid2
    echo python eval_model.py ${models[$i]}
    python eval_model.py ${models[$i]}
done