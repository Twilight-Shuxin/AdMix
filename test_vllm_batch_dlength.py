import sys
import time
import copy
import math
import random
import asyncio
from tqdm import tqdm
from llm_utils import get_data_from_vllm, async_inference
from profiling.constants import write_list_to_csv

MAX_PROMPT_LENGTH = 3000
MIN_PROMPT_LENGTH = 5
TOKEN_BUDGET = 2200
random.seed(42)

def search_configs(total, avg_length, enlarge = 200):
    avg_config = [avg_length] * total
    configs = [avg_config]
    for _ in range(enlarge):
        config = avg_config.copy()
        random_change = 200
        for _ in range(random_change):
            id_1 = random.randint(0, total - 1)
            id_2 = random.randint(0, total - 1)
            if(id_1 == id_2):
                if(id_1):
                    id_1 -= 1
                else: 
                    id_1 += 1
            original_1 = config[id_1]
            original_2 = config[id_2]
            max_change = min(MAX_PROMPT_LENGTH - original_1, original_2 - MIN_PROMPT_LENGTH)
            if(max_change > 0):
                change = random.randint(1, max_change)
                config[id_1] += change
                config[id_2] -= change
        configs.append(config)
    return configs

async def test(total, avg_length, configs):
    all_data = []
    data_prefix = []
    out_len = math.ceil(total * avg_length / TOKEN_BUDGET) + 20
    for config in tqdm(configs):
        decode_tasks = []
        for idx in range(total):
            decode_tasks.append(asyncio.create_task(async_inference(config[idx], out_len)))
        await asyncio.gather(*decode_tasks)
        additional_data, data = get_data_from_vllm()
        data_prefix.append([len(all_data), config])
        all_data.extend(data)
    return all_data, data_prefix

def main():
    decode_tkn = 10
    avg_length = 1550
    configs = search_configs(decode_tkn, avg_length)
    data, data_description = asyncio.run(test(decode_tkn, avg_length, configs))
    data.sort()
    print("decode tkn:", decode_tkn, "avg_length:", avg_length)
    write_list_to_csv("./batch_predict/inputs/" + sys.argv[1], ["Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn", "Latency"], data)

if __name__ == "__main__":
    main()