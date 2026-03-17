import sys
import asyncio
import random
from tqdm import tqdm
from llm_utils import get_data_from_vllm, async_inference
from profiling.constants import write_list_to_csv

KV_CACHE_TKN_SPACE = 86400
TKN_BUDGET = 2400
MAX_PROMPT_LENGTH = 3000

POISSON_RATE = 0.4    # avg requests/sec for background Poisson traffic
TEST_TIME = 300    # wall-clock time limit in seconds

MIN_OUTPUT_LEN = 50
MAX_OUTPUT_LEN = 500
TOTAL_REQS = 120       # max concurrent requests across all generators


class SharedCounter:
    """Mutable integer counter shared across coroutines.

    Uses __iadd__/__isub__ returning self so that `counter += n` in any
    coroutine (including async_inference) mutates the shared object rather
    than rebinding a local variable.
    """
    def __init__(self, value=0):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self

    def __isub__(self, other):
        self.value -= other
        return self

    def __lt__(self, other):
        return self.value < other

    def __repr__(self):
        return str(self.value)


_cached_counter = SharedCounter(0)
_active_reqs = 0
_finished_queue: asyncio.Queue = None


async def _limited_inference(*args, **kwargs):
    """Run inference only if active request count is below TOTAL_REQS; drop otherwise."""
    global _active_reqs
    if _active_reqs >= TOTAL_REQS:
        return
    _active_reqs += 1
    try:
        await async_inference(*args, **kwargs)
    finally:
        _active_reqs -= 1


async def tracked_inference(prompt_len, output_len):
    """Run inference and track its contribution to the cached token count."""
    if not (_cached_counter < KV_CACHE_TKN_SPACE):
        return
    await _limited_inference(prompt_len, output_len, cached_counter=_cached_counter)
    await _finished_queue.put(prompt_len + output_len)


async def _cache_inference(prompt_len, output_len):
    """Issue cache-filling request(s) for prompt_len, then put prompt_len+output_len to _finished_queue.

    If prompt_len > MAX_PROMPT_LENGTH, splits into concurrent fire-and-forget sub-requests.
    """
    if prompt_len > MAX_PROMPT_LENGTH:
        n = (prompt_len + MAX_PROMPT_LENGTH - 1) // MAX_PROMPT_LENGTH
        per_out = max(1, output_len // n)
        for pl in _split_tokens(prompt_len, n):
            asyncio.create_task(_limited_inference(pl, per_out, cached_counter=_cached_counter))
    else:
        await _limited_inference(prompt_len, output_len, cached_counter=_cached_counter)
    await _finished_queue.put(prompt_len + output_len)


async def cache_builder(stop_event):
    tasks = []
    try:
        await asyncio.sleep(60)
        while not stop_event.is_set():
            total = await _finished_queue.get()
            await asyncio.sleep(0.1)
            if _cached_counter < KV_CACHE_TKN_SPACE:
                o = random.randint(100, MAX_OUTPUT_LEN)
                tasks.append(asyncio.create_task(_cache_inference(total, o)))
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _split_tokens(total, n):
    """Split total into n positive integers summing to total."""
    if n == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), n - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


async def prefill_burst(stop_event):
    """
    Independently issue bursts of prefill-heavy requests (output_len=1).

    Each iteration:
      1. Pick a random total token budget in [5, TKN_BUDGET).
      2. Split it across a random number of concurrent requests.
      3. Fire all requests simultaneously (output_len=1 each) and await completion.
      4. Repeat until stop_event is set.
    """
    while not stop_event.is_set():
        total = random.randint(5, TKN_BUDGET - 1)
        n = random.randint(1, min(total, 12))
        prompt_lens = _split_tokens(total, n)
        tasks = [
            asyncio.create_task(_limited_inference(pl, 1))
            for pl in prompt_lens
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


async def poisson_requests(stop_event):
    """Issue random requests following a Poisson arrival process."""
    while not stop_event.is_set():
        await asyncio.sleep(random.expovariate(POISSON_RATE))
        if stop_event.is_set():
            break
        p = random.randint(50, TKN_BUDGET // 2)
        o = random.randint(50, TKN_BUDGET - p)
        asyncio.create_task(tracked_inference(p, o))


async def test():
    global _cached_counter, _active_reqs, _finished_queue
    _cached_counter = SharedCounter(0)
    _active_reqs = 0
    _finished_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    builder = asyncio.create_task(cache_builder(stop_event))
    poisson = asyncio.create_task(poisson_requests(stop_event))
    prefill = asyncio.create_task(prefill_burst(stop_event))

    await asyncio.sleep(TEST_TIME)
    stop_event.set()
    builder.cancel()
    poisson.cancel()
    prefill.cancel()
    await asyncio.gather(builder, poisson, prefill, return_exceptions=True)


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "kv_cache_fill.csv"
    for _ in tqdm(range(10)):
        asyncio.run(test())
    additional_data, data = get_data_from_vllm()
    data.sort()
    write_list_to_csv(
        f"./batch_predict/inputs/{output_file}",
        ["Cached Tkns", "Scheduled Tkns", "Latency"],
        data,
    )


if __name__ == "__main__":
    main()
