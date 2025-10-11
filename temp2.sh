#!/bin/bash

python test_llm.py --model llm --scheduler predictive --trace LLM[13] --device 1 --path data/test_trace >out.out 2>&1
python test_llm.py --model llm --scheduler predictive --trace LLM[14] --device 1 --path data/test_trace >>out.out 2>&1
python test_llm.py --model llm --scheduler predictive --trace LLM[15] --device 1 --path data/test_trace >>out.out 2>&1