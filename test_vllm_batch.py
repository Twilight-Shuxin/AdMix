import sys
import time
import random
import asyncio
from llm_utils import get_data_from_vllm, async_inference
from profiling.constants import write_list_to_csv

DECODE_PROMPT_LEN = 100
TOKEN_BUDGET = 2200
MAX_PREFILL_CNT = 20
PREFILL_PROMPT_LENGTHS = [2000, 1000, 500, 100, 50]
PREFILL_REQ_OUTLEN = 2

def search_configs(token_budget, config, idx):
    if(idx >= len(PREFILL_PROMPT_LENGTHS)):
        return [config]
    if(len(config) == MAX_PREFILL_CNT):
        return [config]
    configs = []
    configs.extend(search_configs(token_budget, config, idx + 1))
    prompt_len =  PREFILL_PROMPT_LENGTHS[idx]
    if(prompt_len < token_budget):
        configs.extend(search_configs(token_budget - prompt_len, config + [prompt_len], idx))
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
        task_list.append(asyncio.create_task(async_inference(max(1, length + random.randint(-20, 20)), PREFILL_REQ_OUTLEN)))
    return task_list

async def test(decode_cnt, prefill_configs):
    out_len = 6 * len(prefill_configs)
    first_generated_barrier = CounterBarrier(decode_cnt)
    decode_tasks = []
    for idx in range(decode_cnt):
        decode_tasks.append(asyncio.create_task(async_inference(DECODE_PROMPT_LEN, out_len, first_generated_barrier)))
    await first_generated_barrier.wait()
    for config in prefill_configs:
        for _ in range(2):
            prefill_tasks = prefill_test(config)
            await asyncio.gather(*prefill_tasks)
    await asyncio.gather(*decode_tasks)

def main():
    for idx, decode_tkn in enumerate(range(0, 151, 20)):
        prefill_tkn_budge = TOKEN_BUDGET - decode_tkn
        prefill_configs = search_configs(prefill_tkn_budge, [], 0)
        start_time = time.perf_counter()
        asyncio.run(test(decode_tkn, prefill_configs))
        print("Running: ", idx, decode_tkn, len(prefill_configs), str(round(time.perf_counter() - start_time, 3)) + "s")
        
    additional_data, data = get_data_from_vllm()
    write_list_to_csv("./data/" + sys.argv[1],["Latency", "Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn"], data)

if __name__ == "__main__":
    main()