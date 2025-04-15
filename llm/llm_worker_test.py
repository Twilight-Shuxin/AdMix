from profiling.models.llm_worker import LLMWorker
from dataset.dataset import Dataset
import requests
import json
import time

class LLMWorkerTest(LLMWorker):
    def __init__(self, device_id, mps, pipe, barrier, log_path):
        super().__init__(device_id, mps, pipe, barrier, log_path)
        self.first_time = []
        self.all_tokens_time = []

    def seq_inference(self):
        generation_input = {
            "prompt": "my name is alice how about you",
            "stream": self.server_config['stream'],
            "ignore_eos": False,
            "max_tokens": 105,
            "temperature": self.server_config['temperature'],
        }
        session = requests.Session()
        session.trust_env = False
        start_time = time.time()
        response = session.post(
            f"http://{self.server_config['host']}:{self.server_config['port']}/generate",
            data = json.dumps(generation_input),
        )
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return
        first_chunk_received = False
        for chunk in response.iter_lines():
            buffer = chunk
            # If this is the first chunk, record the time taken
            if not first_chunk_received:
                first_chunk_time = time.perf_counter() - start_time
                first_chunk_received = True
        output = json.loads(buffer.decode("utf-8"))
        # print(output)
    
    def init_start(self):
        # self.start_server()
        self.dataset = Dataset("shareGPT.json")

    def run(self) -> None:
        print("hi")
        self.init_start()
        self.barrier.wait()
        self.seq_inference()
        # self.stop_server()