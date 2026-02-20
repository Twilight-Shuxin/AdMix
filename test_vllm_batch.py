import sys
import time
import copy
import random
import asyncio
from tqdm import tqdm
from llm_utils import get_data_from_vllm, async_inference
from profiling.constants import write_list_to_csv

DECODE_PROMPT_LEN = 100
TOKEN_BUDGET = 2200
MAX_PREFILL_CNT = 20
PREFILL_PROMPT_LENGTHS = [2000, 1000, 500, 100, 50]
PREFILL_PROMPT_LENGTH_DIFF_HALF = [500, 250, 200, 25, 20]
PREFILL_REQ_OUTLEN = 2
SMALL_LENGTH_MAX = 20
CONFIG_REPEAT = 2

def search_configs(token_budget, configs, idx):
    if(idx >= len(PREFILL_PROMPT_LENGTHS)):
        return copy.deepcopy(configs)
    if(len(configs[0]) == MAX_PREFILL_CNT):
        return copy.deepcopy(configs)
    collected_configs = []
    collected_configs.extend(search_configs(token_budget, configs, idx + 1))
    prompt_len =  PREFILL_PROMPT_LENGTHS[idx]
    if(prompt_len < token_budget):
        for i in range(CONFIG_REPEAT):
            real_prompt_len = max(1, prompt_len + random.randint(-PREFILL_PROMPT_LENGTH_DIFF_HALF[idx], PREFILL_PROMPT_LENGTHS[idx - 1]))
            configs[i].append(real_prompt_len)
        collected_configs.extend(search_configs(token_budget - prompt_len, configs, idx))
        for i in range(CONFIG_REPEAT):
            configs[i].pop()
    return collected_configs

def small_len_configs():
    configs = []
    rep = 5
    for i in range(1, MAX_PREFILL_CNT + 6):
        for _ in range(rep):
            configs.append([random.randint(1, SMALL_LENGTH_MAX) for _ in range(i)])
    return configs

class CounterBarrier:
    def __init__(self, n_tasks: int):
        self.n = n_tasks
        self.count = 0
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def signal(self):
        async with self.lock:
            self.count += 1
            if self.count == self.n:
                self.event.set()

    async def wait(self):
        if(self.n > 0):
            await self.event.wait()

def prefill_test(prefill_config):
    task_list = []
    for length in prefill_config:
        task_list.append(asyncio.create_task(async_inference(length, PREFILL_REQ_OUTLEN)))
    return task_list

async def test(decode_cnt, prefill_configs):
    out_len = max(2 * len(prefill_configs), 4000)
    first_generated_barrier = CounterBarrier(decode_cnt)
    decode_tasks = []
    for idx in range(decode_cnt):
        decode_tasks.append(asyncio.create_task(async_inference(DECODE_PROMPT_LEN, out_len, first_generated_barrier)))
    await first_generated_barrier.wait()
    for config in tqdm(prefill_configs):
        prefill_tasks = prefill_test(config)
        await asyncio.gather(*prefill_tasks)
        time.sleep(0.03)
    await asyncio.gather(*decode_tasks)

def main():
    for idx, decode_tkn in enumerate(range(0, 121, 20)):
        prefill_tkn_budge = TOKEN_BUDGET - decode_tkn
        prefill_configs = search_configs(prefill_tkn_budge, [[] for _ in range(CONFIG_REPEAT)], 0)
        prefill_configs.extend(small_len_configs())
        random.shuffle(prefill_configs)
        print("Number of configs:", len(prefill_configs))
        start_time = time.perf_counter()
        asyncio.run(test(decode_tkn, prefill_configs))
        print("Running: ", idx, decode_tkn, str(round(time.perf_counter() - start_time, 3)) + "s")
        
    additional_data, data = get_data_from_vllm()
    data.sort()
    write_list_to_csv("./batch_predict/inputs/" + sys.argv[1], ["Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn", "Latency"], data)

if __name__ == "__main__":
    main()