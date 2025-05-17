#!/bin/bash

# python test_llm.py --model llm --profile_group 0 --seqlen 10,10 --trace BurstGPT_1.csv --device 1
# echo =========== python test_llm.py --model llm --profile_group 0,3,4 --trace BurstGPT_1.csv --device 1
# python test_llm.py --model llm --profile_group 0 --trace BurstGPT_1.csv --device 1 --test_time 30
# python test_llm.py --model llm --profile_group 0,3,4 --trace BurstGPT_1.csv --device 1

models=("resnet50" "vgg19" "inception_v3")
# batch_sizes=(1 2 4 8 16 32)
batch_sizes=(1 2 4 8)

for i in "${!models[@]}"; do 
    for batch_size in "${batch_sizes[@]}"; do
        echo =========== python test_llm.py --model ${models[$i]},llm --batch_size $batch_size --profile_group 0,3,4 --trace ,BurstGPT_1.csv --device 1
        python test_llm.py --model ${models[$i]},llm --batch_size $batch_size --profile_group 0,3,4 --trace ,BurstGPT_1.csv --device 1
    done
done

echo =========== python test_llm.py --model vgg19,llm --batch_size 32 --profile_group 0,3,4 --mps 30,100 --trace ,BurstGPT_1.csv --device 1
python test_llm.py --model vgg19,llm --batch_size 32 --profile_group 0,3,4 --mps 30,100 --trace ,BurstGPT_1.csv --device 1
echo =========== python test_llm.py --model vgg19,llm --batch_size 32 --profile_group 0,3,4 --mps 40,100 --trace ,BurstGPT_1.csv --device 1
python test_llm.py --model vgg19,llm --batch_size 32 --profile_group 0,3,4 --mps 40,100 --trace ,BurstGPT_1.csv --device 1
echo =========== python test_llm.py --model inception_v3,llm --batch_size 32 --profile_group 0,3,4 --mps 30,100 --trace ,BurstGPT_1.csv --device 1
python test_llm.py --model inception_v3,llm --batch_size 32 --profile_group 0,3,4 --mps 30,100 --trace ,BurstGPT_1.csv --device 1
echo =========== python test_llm.py --model inception_v3,llm --batch_size 32 --profile_group 0,3,4 --mps 40,100 --trace ,BurstGPT_1.csv --device 1
python test_llm.py --model inception_v3,llm --batch_size 32 --profile_group 0,3,4 --mps 40,100 --trace ,BurstGPT_1.csv --device 1
