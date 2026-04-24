import os
import sys
import time
import signal
import asyncio
import argparse
import requests
import subprocess
import torch
from trace.trace_utils import LLMTrace, set_LLM_trace_config, read_LLM_trace_config
from llm_utils import get_data_from_llm, async_inference
from profiling.constants import write_list_to_csv

# python test_vllm_batch_trace.py my_trace_config output.csv --platform vllm --model qwen14b --test-time 120

_LLM_MODEL_MAP = {
    "llama7b": "meta-llama/Llama-2-7b-chat-hf",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
}

_TP_MAP = {
    "llama7b": 1,
    "qwen14b": 2,
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

async def test_trace(trace):
    start_time_pointer = time.perf_counter()
    test_time = trace.trace_config.test_time
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
    for task in task_list:
        if not task.done():
            task.cancel()
    await asyncio.gather(*task_list, return_exceptions=True)

def get_trace(trace_config_id, test_time=None):
    set_LLM_trace_config(trace_config_id)
    trace_config = read_LLM_trace_config()
    if test_time:
        trace_config.test_time = test_time
    return LLMTrace(trace_config)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_config_id")
    parser.add_argument("output_file")
    parser.add_argument("--platform", required=True, choices=["vllm", "sglang"], help="LLM serving platform")
    parser.add_argument("--model", required=True, help="Model name or shorthand (llama7b, qwen14b, qwen32b, or full HF path)")
    parser.add_argument("--test-time", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    llm_model = _LLM_MODEL_MAP.get(args.model, args.model)
    os.environ['ADMIX_LLM_SERVING_PLATFORM'] = args.platform

    server_proc = start_server(args.platform, args.model, llm_model)
    try:
        additional_data, processed, data = get_data_from_llm()
        loop = asyncio.new_event_loop()
        trace = get_trace(args.trace_config_id, args.test_time)
        loop.run_until_complete(test_trace(trace))
        additional_data, processed, data = get_data_from_llm()
        data.sort(key=lambda x: (x[0] * x[1]))
        data.sort()
        write_list_to_csv(f"./batch_predict/inputs/{args.output_file}", ["Scheduled Tkns", "Cached Tkns", "Latency"], data)
    finally:
        stop_server(server_proc)

if __name__ == "__main__":
    main()
