import sys
import asyncio
import aiohttp
from profiling.constants import *
from profiling.scheduling.scheduler import get_data_length
from profiling.scheduling.monitor import LLM_INTENSITY_SOCKET_PATH
from dataset.sharegpt_dataset import ShareGPTDataset
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config

server_config = {
    'host': "localhost",
    "port": 8000,
    "temperature": 0,
    "stream": True
}
sharegpt_dataset = ShareGPTDataset("shareGPT.json")

def get_prompt(prompt_len):
    prompt, _ = sharegpt_dataset.get_query(InputConfig(seq_len = prompt_len))
    return prompt

async def async_inference(input_len, output_len):
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
                buffer = chunk

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

def main():
    loop = asyncio.new_event_loop()

    trace_config_id = int(sys.argv[1])
    test_time = None
    if(len(sys.argv) > 2):
        test_time = int(sys.argv[2])
    
    set_LLM_trace_config(trace_config_id)
    trace_config = read_LLM_trace_config()
    if(not (test_time and hasattr(trace_config, 'test_time'))):
        if(hasattr(trace_config, 'test_time')):
            test_time = trace_config.test_time
    trace = LLMTrace(trace_config)
    loop.run_until_complete(test_trace(trace, test_time))

    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_socket.connect(LLM_INTENSITY_SOCKET_PATH)
    client_socket.sendall(b"hi")
    length = get_data_length(client_socket)
    data = client_socket.recv(length)
    data = json.loads(data.decode("utf-8"))
    client_socket.close()
    shuxin_csv(["Latency", "Total kn", "Decode Tkn", "Decode Len", "Prefill Cnt", "Prefill Tkn", "Prefill Reqs"], data)


if __name__ == "__main__":
    main()