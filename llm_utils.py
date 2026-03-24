import os
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

def build_generation_input(prompt, output_len):
    server_type = os.environ.get('ADMIX_LLM_SERVING_PLATFORM', 'vllm').lower()
    if server_type == 'sglang':
        return {
            "text": prompt,
            "stream": server_config['stream'],
            "sampling_params": {
                "max_new_tokens": output_len,
                "temperature": server_config['temperature'],
                "ignore_eos": True,
            },
        }
    else:  # vllm
        return {
            "prompt": prompt,
            "stream": server_config['stream'],
            "ignore_eos": True,
            "max_tokens": output_len,
            "temperature": server_config['temperature'],
        }

sharegpt_dataset = ShareGPTDataset("shareGPT.json")

def get_prompt(prompt_len):
    prompt, _ = sharegpt_dataset.get_query(InputConfig(seq_len = prompt_len))
    return prompt

async def async_inference(input_len, output_len):
    prompt = get_prompt(input_len)
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    start = time.perf_counter()
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = build_generation_input(prompt, output_len)
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
        

def get_data_from_llm():
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_socket.connect(LLM_INTENSITY_SOCKET_PATH)
    client_socket.sendall(b"hi")
    length = get_data_length(client_socket)
    data = conn_get_data_chunked(client_socket, length)
    data = json.loads(data.decode("utf-8"))
    client_socket.close()
    return data