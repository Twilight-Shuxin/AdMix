import sys
import time
import asyncio
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config
from llm_utils import get_data_from_vllm, async_inference
from profiling.constants import write_list_to_csv

async def test_trace(trace, test_time):
    start_time_pointer = time.perf_counter()
    time_limit = test_time + time.perf_counter()
    task_list = []
    while(time.perf_counter() < time_limit):
        req_time, sample_in_length, sample_out_length = trace.next()
        sleep_time = min(req_time, test_time) + start_time_pointer - time.perf_counter()
        if(sleep_time > 0):
            await asyncio.sleep(sleep_time)
        if(req_time >= test_time):
            break
        task_list.append(asyncio.create_task(async_inference(sample_in_length, sample_out_length)))
    for id, task in enumerate(task_list):
        if(not task.done()):
            task.cancel()
    await asyncio.gather(*task_list, return_exceptions=True)

def get_trace(trace_config_id, test_time = None):    
    set_LLM_trace_config(trace_config_id)
    trace_config = read_LLM_trace_config()
    if(not (test_time and hasattr(trace_config, 'test_time'))):
        if(hasattr(trace_config, 'test_time')):
            test_time = trace_config.test_time
    return LLMTrace(trace_config)

def main():
    loop = asyncio.new_event_loop()
    trace_config_id = int(sys.argv[2])
    test_time = int(sys.argv[3]) if len(sys.argv) > 3 else None
    trace = get_trace(trace_config_id, test_time)
    loop.run_until_complete(test_trace(trace, test_time))
    data = get_data_from_vllm()
    write_list_to_csv("./data/" + sys.argv[1],["Latency", "Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn"], data)

if __name__ == "__main__":
    main()