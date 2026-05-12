import sys
import os
import math
import time
import random
import asyncio
import argparse
from tqdm import tqdm
from itertools import chain, cycle
import matplotlib.pyplot as plt
from llm_utils import get_data_from_llm, async_inference, _LLM_MODEL_MAP, start_server, stop_server
from profiling.constants import write_list_to_csv

# 7b: 86400
KV_CACHE_TKN_SPACE = 152800
MAX_PROMPT_LENGTH = 3000
TKN_BUDGET = 2020

MIN_OUTPUT_LEN = 50
MAX_OUTPUT_LEN = 3000
TEST_TIME = 4000  # wall-clock time limit in seconds

PROMPT_LEN = 5
MINN = math.ceil(KV_CACHE_TKN_SPACE / (MAX_OUTPUT_LEN + PROMPT_LEN))

def _split_tokens(total, n):
    if n == 1:
        return [min(total, MAX_PROMPT_LENGTH)]
    cuts = sorted(random.sample(range(1, total), n - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    return [min(p, MAX_PROMPT_LENGTH) for p in parts]


async def prefill_burst(stop_event, limit):
    while not stop_event.is_set():
        total = random.randint(50, limit)
        n = random.randint(1, min(total, 12))
        prompt_lens = _split_tokens(total, n)
        tasks = [
            asyncio.create_task(async_inference(pl, 1))
            for pl in prompt_lens
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

async def _test(k, MAX_PROMPT_LENGTH_TMP):
    stop_event = asyncio.Event()
    burst = asyncio.create_task(prefill_burst(stop_event, MAX_PROMPT_LENGTH_TMP))
    if(k):
        output_len = max(min(3000, KV_CACHE_TKN_SPACE // k - PROMPT_LEN), 1)
        tasks = [asyncio.create_task(async_inference(PROMPT_LEN, output_len)) for _ in range(k)]
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        await asyncio.sleep(300)
    stop_event.set()
    await burst

async def test():
    await _test(0, 400)
    deadline = time.monotonic() + TEST_TIME
    k_iter = cycle(range(1, MINN + 10, 5))
    last_t = 0.0
    with tqdm(desc="Sweeping k") as pbar:
        while time.monotonic() < deadline:
            k = next(k_iter)
            for prompt_limit in (500, TKN_BUDGET):
                pbar.set_postfix(k=k, last=f"{last_t:.1f}s")
                t0 = time.monotonic()
                await _test(k, prompt_limit)
                last_t = time.monotonic() - t0
                pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", nargs="?", default="kv_cache_fill_new.csv")
    parser.add_argument("--platform", required=True, choices=["vllm", "sglang"], help="LLM serving platform")
    parser.add_argument("--model", required=True, help="Model name or shorthand (llama7b, qwen14b, qwen32b, or full HF path)")
    return parser.parse_args()

def main():
    args = parse_args()
    llm_model = _LLM_MODEL_MAP.get(args.model, args.model)
    os.environ['ADMIX_LLM_SERVING_PLATFORM'] = args.platform

    server_proc = start_server(args.platform, args.model, llm_model)
    try:
        additional_data, processing, data = get_data_from_llm()
        asyncio.run(test())
        additional_data, processing, data = get_data_from_llm()

        # Plot Cached Tkns over time (request order)
        cached_tkns = [row[1] for row in data]
        plt.figure(figsize=(12, 5))
        plt.plot(cached_tkns)
        plt.xlabel("Request Index")
        plt.ylabel("Cached Tkns")
        plt.title("Cached Tkns over Time")
        plt.tight_layout()
        plot_name = os.path.splitext(args.output_file)[0] + "_cached_tkns.png"
        plt.savefig(f"./batch_predict/plot/outputs/{plot_name}", dpi=150, bbox_inches="tight")
        plt.close()

        data.sort()
        write_list_to_csv(
            f"./batch_predict/inputs/{args.output_file}",
            ["Scheduled Tkns", "Cached Tkns", "Latency"],
            data,
        )
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()
