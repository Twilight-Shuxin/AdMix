from profiling.models.split.inception import inception_v3
from profiling.models.model_worker import ModelWorker
from dataset.dataset import Dataset
from trace.trace_utils import BurstGPTTrace
import torch
import time
from profiling.constants import *
from trace.trace_utils import Trace
import asyncio
import aiohttp
import json

class LLMWorker(ModelWorker):
    def __init__(self, mps, pipe, barrier, log_path):
        super().__init__("llm", mps, pipe, barrier, log_path)
        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True
        }
        self.logs = []
        self.SPEED_CONST = 0.25
    
    def init_start(self):
        self.dataset = Dataset("shareGPT.json")
    
    def get_input(self, test_config):
        return self.dataset.get_query(test_config)[0]
    
    def get_trace(self, trace_path):
        self.trace = BurstGPTTrace(trace_path)
    
    def warmup(self):
        warmup_config = TestConfig(1, None, 1, 0, self.total_module, -1, [5,5])
        warmup_config.fit(self)
        self.random_test_period(warmup_config, 5)
    
    def write_log(self, test_config):
        self.write_analytics("_".join([test_config.tag, "ft", "summary.log"]), [x[0] for x in self.logs])
        self.write_analytics("_".join([test_config.tag, "at", "summary.log"]), [x[1] for x in self.logs])
        log_filepath = os.path.join(self.log_path, "_".join([test_config.tag, "all.log"]))  
        with open(log_filepath, "w+") as log_output:
            for log_record in self.logs: 
                log_output.write(list_to_string(log_record) + "\n")
    
    async def _inference(self, input, rec, max_tokens = 1024):
        # no start & end available for LLMworker served by vllm
        timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
        send_time = time.time()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            generation_input = {
                "prompt": "my name is alice", #input,
                "stream": self.server_config['stream'],
                "ignore_eos": False,
                "max_tokens": 5,
                "temperature": self.server_config['temperature'],
            }
            first_chunk_time = 0
            reading_time = 0
            start_time = time.perf_counter()
            user_idle_time = 0
            async with session.post(
                f"http://{self.server_config['host']}:{self.server_config['port']}/generate", json=generation_input
            ) as resp:
                if resp.status != 200:
                    print(f"Error: {resp.status} {resp.reason}")
                    print(await resp.text())
                    return None, None, None

                if self.server_config['stream']:
                    buffer = b""
                    first_chunk_received = False
                    async for chunk in resp.content.iter_any():
                        buffer = chunk
                        # If this is the first chunk, record the time taken
                        if not first_chunk_received:
                            first_chunk_time = time.perf_counter() - start_time
                            first_chunk_received = True
                        user_idle_time += max(0, time.perf_counter() - start_time - reading_time)
                        reading_time += self.SPEED_CONST
                    output = json.loads(buffer.decode("utf-8"))  # Decode JSON
                else:
                    output = await resp.json()      
                end_time = time.perf_counter()
                total_chunk_time = end_time - start_time
        if(rec):
            self.logs.append([first_chunk_time, total_chunk_time, user_idle_time])

    def inference(self, input, rec = False):
        self.loop.run_until_complete(self._inference(input, rec))

    def random_test_period(self, test_config, time_len, rec = False):
        start_time = time.time()
        if(rec):
            pids = self.start_dcgm(test_config)
        while(time.time() - start_time < time_len):
            input = self.get_input(test_config)
            self.inference(input, rec)
        if(rec):
            stop_dcgm(pids)

    async def trace_test_period(self, test_config, time_len, rec = False):
        if(rec):
            pids = self.start_dcgm(test_config)
        test_start_time = time.time()
        time_limit = test_start_time + time_len
        cnt = 0
        task_list = []
        while(True):
            req_time, sample_in_length, sample_out_length = self.trace.next()
            test_config.seq_len = [sample_in_length, sample_out_length]
            sleep_time = req_time + test_start_time - time.time()
            if(sleep_time > 0):
                time.sleep(sleep_time)
            if(time.time() > time_limit):
                break
            input = self.get_input(test_config)
            task_list.append(asyncio.create_task(self._inference(input, rec, sample_out_length)))
            cnt += 1
        print("Processed: ", cnt)
        await asyncio.gather(*task_list)
        if(rec):
            stop_dcgm(pids)
            self.logs = sorted(self.logs, key = lambda x: x[0]) 
    
    def common_start(self):
        self.total_module = -1
    
    def random_test(self, test_config):
        self.random_test_period(test_config, 10)
        if(test_config.trace_path != None):
            asyncio.run(self.trace_test_period(test_config, 40, True))
        else:
            self.random_test_period(test_config, 40, True)
        self.random_test_period(test_config, 5)
        self.write_log(test_config)

    def run(self) -> None:
        self.loop = asyncio.get_event_loop()
        self.init_start()
        self.common_start()
        self.warmup()
        self.barrier.wait()
        (action) = self.pipe.recv()
        if(action == "serve"):
            self.serve()
        elif(action == "test"):
            test_config = self.pipe.recv()
            test_config.fit(self)
            self.random_test(test_config)
        else:
            raise NotImplementedError
    

