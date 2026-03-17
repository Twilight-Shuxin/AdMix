import json
import socket
import asyncio
import aiohttp
import time
from profiling.constants import *
from dataset.sharegpt_dataset import ShareGPTDataset
from profiling.scheduling.monitor import LLM_INTENSITY_SOCKET_PATH
from profiling.scheduling.scheduler import get_data_length, conn_get_data_chunked

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

async def async_inference(input_len, output_len, cached_counter = None):
    cnt = input_len
    try:
        prompt = get_prompt(input_len)
        timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
        start = time.perf_counter()
        
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
                    cnt += 1
                    if(first_generated_barrier and (not buffer)):
                        await first_generated_barrier.signal()
                    if(not buffer):
                        cached_counter += input_len + 1
                    else:
                        cached_counter += 1
                    if(shared_queue is not None):
                        shared_queue.put_nowait(1)
                    buffer = chunk
                if(shared_queue is not None):
                    shared_queue.put_nowait(-1)
    except asyncio.CancelledError:
        cached_counter -= cnt
        

def get_data_from_vllm():
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_socket.connect(LLM_INTENSITY_SOCKET_PATH)
    client_socket.sendall(b"hi")
    length = get_data_length(client_socket)
    data = conn_get_data_chunked(client_socket, length)
    data = json.loads(data.decode("utf-8"))
    client_socket.close()
    return data