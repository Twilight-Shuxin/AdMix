#!/bin/bash

# echo python test_llm.py --model llm --scheduler predictive --trace LLM[16] --device 1
# python test_llm.py --model llm --scheduler predictive --trace LLM[16] --device 1  >>out.out
# python test_llm.py --model llm --scheduler predictive --trace LLM[14] --device 1 --path data/test_trace >>out.out 2>&1
# python test_llm.py --model llm --scheduler predictive --trace LLM[15] --device 1 --path data/test_trace >>out.out 2>&1

# echo python test_llm.py --model llm,vgg19 --trace LLM[17], --path data/test1  --scheduler predictive --test_time 5000 --device 1
# python test_llm.py --model llm,vgg19 --trace LLM[17], --path data/test1  --scheduler predictive --test_time 5000 --device 1 >out.out

# echo python test_llm.py --model llm --trace LLM[16] --scheduler AIMD --path data/tmp --device 1
# python test_llm.py --model llm --trace LLM[16] --scheduler AIMD --path data/tmp --device 1 >>out.out

# echo python test_llm.py --model llm,unet3d --trace LLM[16], --scheduler AIMD --path data/tmp --profile_group 0 --device 1
# python test_llm.py --model llm,unet3d --trace LLM[16],  --scheduler AIMD --path data/tmp --profile_group 0 --device 1 >>out.out

# echo python test_llm.py --model llm,unet3d --trace LLM[16], --scheduler predictive_v2 --profile_group 0 --device 1
# python test_llm.py --model llm,unet3d --trace LLM[16],  --scheduler predictive_v2 --profile_group 0 --device 1 >>out.out

# echo python test_llm.py --model llm,vgg19 --trace LLM[16], --scheduler predictive_v2 --profile_group 0 --device 1
# python test_llm.py --model llm,vgg19 --trace LLM[16],  --scheduler predictive_v2 --profile_group 0 --device 1 >>out.out

# echo python test_llm.py --model llm,vgg19 --trace LLM[16], --scheduler AIMD --path data/tmp --profile_group 0 --device 1
# python test_llm.py --model llm,vgg19 --trace LLM[16],  --scheduler AIMD --path data/tmp --profile_group 0 --device 1 >>out.out


# echo python test_llm.py --model llm --trace LLM[16] --path data/util  --profile_group 0 --device 1
# python test_llm.py --model llm --trace LLM[16] --path data/util  --profile_group 0 --device 1 >>out.out

# echo python test_llm.py --model llm --trace LLM[17] --path data/util  --profile_group 0 --device 1
# python test_llm.py --model llm --trace LLM[17] --path data/util  --profile_group 0 --device 1 >>out.out