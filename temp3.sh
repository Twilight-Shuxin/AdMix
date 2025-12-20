#!/bin/bash

# for id in {0..10}; do   
#     echo python test_llm.py --model llm --scheduler predictive --trace BurstGPT_1.csv[$id] --device 1 --path data/test_trace
#     # python test_llm.py --model llm --scheduler predictive --trace BurstGPT_1.csv[$id] --device 1 --path data/test_trace
# done

# python test_llm.py --model llm,vgg19 --scheduler predictive --trace LLM[13], --test_time 150 --device 1
# echo python test_llm.py --model llm --scheduler AIMD --trace LLM[14] --test_time 7200 --device 1 --path data/aimd >out.out 2>&1
# python test_llm.py --model llm --scheduler AIMD --trace LLM[14] --test_time 7200 --device 1 --path data/aimd >out.out 2>&1
# echo python test_llm.py --model llm --scheduler AIMD --trace LLM[13] --test_time 7200 --device 1 --path data/aimd >out.out 2>&1
# python test_llm.py --model llm --scheduler AIMD --trace LLM[13] --test_time 7200 --device 1 --path data/aimd >out.out 2>&1
# echo python test_llm.py --model llm,vgg19 --scheduler AIMD --trace LLM[13], --test_time 7200 --device 1 --path data/aimd >out.out 2>&1
# python test_llm.py --model llm,vgg19 --scheduler AIMD --trace LLM[13], --test_time 7200 --device 1 --path data/aimd >out.out 2>&1
# echo python test_llm.py --model llm,vgg19 --scheduler AIMD --trace LLM[14], --test_time 7200 --device 1 --path data/aimd >>out.out 2>&1
# python test_llm.py --model llm,vgg19 --scheduler AIMD --trace LLM[14], --test_time 7200 --device 1 --path data/aimd >>out.out 2>&1
echo python test_llm.py --model llm,vgg19 --scheduler predictive --trace LLM[16], --path data/test_trace --device 1 >out.out 2>&1
python test_llm.py --model llm,vgg19 --scheduler predictive --trace LLM[16], --path data/test_trace --device 1 >out.out 2>&1
echo python test_llm.py --model llm,vgg19 --scheduler constant[8] --trace LLM[16], --device 1 --path data/aimd >>out.out 2>&1
python test_llm.py --model llm,vgg19 --scheduler constant[8] --trace LLM[16], --device 1 --path data/aimd >>out.out 2>&1
echo python test_llm.py --model llm --scheduler AIMD --trace LLM[16] --device 1 --path data/aimd >>out.out 2>&1
python test_llm.py --model llm --scheduler AIMD --trace LLM[16] --device 1 --path data/aimd >>out.out 2>&1
echo python test_llm.py --model llm,vgg19 --scheduler AIMD --trace LLM[16], --device 1 --path data/aimd >>out.out 2>&1
python test_llm.py --model llm,vgg19 --scheduler AIMD --trace LLM[16], --device 1 --path data/aimd >>out.out 2>&1