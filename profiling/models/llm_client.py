class LLMClient(ModelClient):
    def __init__(self, model_name, devices, mps, pipe, barrier, log_path, profile_path):
        super().__init__("llm", devices, mps, pipe, barrier, log_path, profile_path)
    
    