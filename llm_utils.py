import os
import json
import time
import signal
import socket
import asyncio
import aiohttp
import requests
import subprocess
import torch
from profiling.constants import *
from dataset.sharegpt_dataset import ShareGPTDataset
from profiling.scheduling.monitor import LLM_INTENSITY_SOCKET_PATH
from profiling.scheduling.scheduler import get_data_length, conn_get_data_chunked

_LLM_MODEL_MAP = {
    "llama7b": "meta-llama/Llama-2-7b-chat-hf",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
}

_TP_MAP = {
    "llama7b": 1,
    "qwen14b": 2,
    "qwen32b": 4,
}

_PORT = 8000

def _wait_for_server(health_url):
    session = requests.Session()
    session.trust_env = False
    print("Starting server...")
    while True:
        try:
            if session.get(health_url).status_code == 200:
                break
        except Exception:
            time.sleep(1)
    print("Server ready!")

def _start_vllm_server(llm_model, tp, memory_util="0.7"):
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    proc = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.api_server",
        "--model", llm_model,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", memory_util,
        "--no-enable-prefix-caching",
    ])
    _wait_for_server(f"http://localhost:{_PORT}/health")
    return proc

def _start_sglang_server(llm_model, tp, memory_util="0.7"):
    env = os.environ.copy()
    env.update({"NO_PROXY": "0.0.0.0,localhost,127.0.0.1", "no_proxy": "0.0.0.0,localhost,127.0.0.1"})
    proc = subprocess.Popen([
        "plugins/sglang_env/.venv/bin/python", "-m", "sglang.launch_server",
        "--model-path", llm_model,
        "--chunked-prefill-size", "2048",
        "--enable-mixed-chunk",
        "--tp", str(tp),
        "--mem-fraction-static", memory_util,
        "--port", str(_PORT),
        "--disable-radix-cache",
        "--log-level", "warning",
        "--disable-overlap-schedule",
    ], env=env)
    _wait_for_server(f"http://localhost:{_PORT}/health")
    return proc

def start_server(platform, model_key, llm_model):
    tp = _TP_MAP.get(model_key, torch.cuda.device_count())
    os.system("bash llm_platform_setup.sh modified")
    if platform == "vllm":
        return _start_vllm_server(llm_model, tp)
    elif platform == "sglang":
        return _start_sglang_server(llm_model, tp)
    raise ValueError(f"Unknown platform: {platform}")

def stop_server(proc):
    if proc:
        proc.send_signal(signal.SIGINT)

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