

class LLMWorker(Process):
    def __init__(self, pipe, worker_barrier, single_worker_barrier):
        self.pipe = pipe
        self.worker_barrier = worker_barrier
        self.single_worker_barrier = single_worker_barrier
        self.device = "cuda"

        self.server_config = {
            'host': "localhost",
            "port": 8000,
            "temperature": 0,
            "stream": True
        }
        self.first_token_slo = 1
        self.idle_time_slo = 0
        self.SPEED_CONST = 0.2
        self.initial_time = self.SPEED_CONST

        self.server_proc = None
        super().__init__()
    
    def init_start(self):
        self.loop = asyncio.get_event_loop()
        self.dataset = ShareGPTDataset("shareGPT.json")
    
    def shutdown(self):
        # self.stop_server()
        return
    
    def start_server(self):
        if(os.environ.get("CUDA_MPS_PIPE_DIRECTORY", None)):
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.true_device_id)
        os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
        memory_util = os.environ.get('GVLLM_KV_MEM', "0.7")
        use_modified = os.environ.get('GVLLM_USE_MODIFIED', 'False').lower() == 'true'
        system_setup_cmd = "bash vllm_setup.sh"
        if(use_modified):
            system_setup_cmd += " modified"
        os.system(system_setup_cmd)
        self.server_proc = subprocess.Popen(["python", "-m", "vllm.entrypoints.api_server"
            , "--model", "meta-llama/Llama-2-7b-chat-hf", "--gpu-memory-utilization", memory_util, "--no-enable-prefix-caching"])
        session = requests.Session()
        session.trust_env = False
        print("Starting server...")
        while(True):
            try:
                response = session.get('http://localhost:8000/health') 
                if(response.status_code == 200):
                    break
            except:
                time.sleep(1)
        print("Server ready!")
    
    def stop_server(self):
        if(self.server_proc):
            self.server_proc.send_signal(signal.SIGINT)
    