import sys
import time
import asyncio
from tqdm import tqdm
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config
from llm_utils import get_data_from_llm, async_inference
from profiling.constants import write_list_to_csv

async def test_trace(trace):
    start_time_pointer = time.perf_counter()
    test_time = trace.trace_config.test_time
    time_limit = test_time + time.perf_counter()
    task_list = []
    # pbar = tqdm(desc="Sending requests", unit="req")
    while(time.perf_counter() < time_limit):
        req_time, sample_in_length, sample_out_length = trace.next()
        sleep_time = min(req_time, test_time) + start_time_pointer - time.perf_counter()
        if(sleep_time > 0):
            await asyncio.sleep(sleep_time)
        if(req_time >= test_time):
            break
        task_list.append(asyncio.create_task(async_inference(sample_in_length, sample_out_length)))
        # pbar.update(1)
    # pbar.close()
    for id, task in enumerate(task_list):
        if(not task.done()):
            task.cancel()
    await asyncio.gather(*task_list, return_exceptions=True)

def get_trace(trace_config_id, test_time = None):    
    set_LLM_trace_config(trace_config_id)
    trace_config = read_LLM_trace_config()
    if(test_time):
        trace_config.test_time = test_time
    return LLMTrace(trace_config)

def main():
    additional_data, processed, data = get_data_from_llm()
    loop = asyncio.new_event_loop()
    test_time = int(sys.argv[2]) if len(sys.argv) > 2 else None
    trace_config_id = sys.argv[1]
    trace = get_trace(trace_config_id, test_time)
    loop.run_until_complete(test_trace(trace))
    additional_data, processed, data = get_data_from_llm()
    data.sort(key=lambda x: (x[0] * x[1]))
    data.sort()
    write_list_to_csv(f"./batch_predict/inputs/vllm_tp2_{trace_config_id}.csv",["Scheduled Tkns", "Cached Tkns", "Latency"], data)

if __name__ == "__main__":
    main()