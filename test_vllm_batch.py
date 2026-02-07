import asyncio
import aiohttp
from profiling.constants import *
from dataset.sharegpt_dataset import ShareGPTDataset

server_config = {
    'host': "localhost",
    "port": 8000,
    "temperature": 0,
    "stream": True
}

DECODE_PROMPT_LEN = 100
TOKEN_BUDGET = 2040
MAX_PREFILL_CNT = 10
PREFILL_PROMPT_LENGTHS = [1900, 1000, 500, 100]
PREFILL_REQ_OUTLEN = 2
sharegpt_dataset = ShareGPTDataset("shareGPT.json")

def search_configs(sum, config, idx):
    if(idx >= len(PREFILL_PROMPT_LENGTHS)):
        return [config]
    if(len(config) == MAX_PREFILL_CNT):
        return [config]
    configs = []
    configs.extend(search_configs(sum, config, idx + 1))
    prompt_len =  PREFILL_PROMPT_LENGTHS[idx]
    if(sum >= prompt_len):
        configs.extend(search_configs(sum - prompt_len, config + [prompt_len], idx))
    return configs

def get_prompt(prompt_len):
    prompt, _ = sharegpt_dataset.get_query(InputConfig(seq_len = prompt_len))
    return prompt

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
        await self.event.wait()

async def async_inference(input_len, output_len, first_generated_barrier = None):
    prompt = get_prompt(input_len)
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "prompt": prompt,
            "stream": server_config['stream'],
            "ignore_eos": True,
            "max_tokens": output_len,
            "temperature": server_config['temperature'],
        }
        async with session.post(
            f"http://{server_config['host']}:{server_config['port']}/generate", json = generation_input
        ) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return
            buffer = None
            async for chunk in resp.content.iter_any():
                if(first_generated_barrier and (not buffer)):
                    await first_generated_barrier.signal()
                buffer = chunk

def prefill_test(prefill_config):
    task_list = []
    for length in prefill_config:
        task_list.append(asyncio.create_task(async_inference(length, PREFILL_REQ_OUTLEN)))
    return task_list

async def test(decode_cnt, prefill_configs):
    out_len = 2 * len(prefill_configs)
    first_generated_barrier = CounterBarrier(decode_cnt)
    decode_tasks = []
    for idx in range(decode_cnt):
        decode_tasks.append(asyncio.create_task(async_inference(DECODE_PROMPT_LEN, out_len, first_generated_barrier)))
    await first_generated_barrier.wait()
    for config in prefill_configs:
        prefill_tasks = prefill_test(config)
        await asyncio.gather(*prefill_tasks)
    await asyncio.gather(*decode_tasks)

def main():
    loop = asyncio.new_event_loop()
    for decode_tkn in range(0, 101, 50):
        if(decode_tkn == 0):
            decode_tkn = 1
        prefill_tkn_budge = TOKEN_BUDGET - decode_tkn
        prefill_configs = search_configs(prefill_tkn_budge, [], 0)
        asyncio.run(test(decode_tkn, prefill_configs))
    loop.run_until_complete(async_inference(3000, PREFILL_REQ_OUTLEN))

if __name__ == "__main__":
    main()