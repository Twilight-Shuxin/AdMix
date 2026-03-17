#!/bin/bash

# nohup ./profile_script.sh 2>&1 &
source ./script_utils.sh
source .venv/bin/activate

models=("resnet50" "vgg19" "unet3d" "deeplabv3" "GAN")
for model in "${models[@]}"; do
    python run.py --model "$model" --devices 1 --mode PROFILE
done