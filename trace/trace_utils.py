import os
import pandas as pd
import json

# Read trace config (full json form) from config file
def read_trace_config_json(filename):
    filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
    with open(filename, 'r') as trace_config_file:
        data = json.load(trace_config_file)
    return data

# Read trace config from config file, only with tag keywords
# trace_type, chunk_size, start_id, max_total_prompt_num, scale
def read_trace_config(filename):
    data = read_trace_config_json(filename)
    return data["trace"], data["chunk_size"], data["start_id"], data["max_prompt_cnt"], data["scale"]  

# Get trace tag from config file
def get_trace_tag():
    trace_type, chunk_size, start_id, max_total_prompt_num, scale = read_trace_config("trace_config.json")
    trace_arg = [trace_type, chunk_size, start_id, max_total_prompt_num, scale]
    trace_tag = ",".join([str(x) for x in trace_arg])
    return trace_tag

# Set trace config file to correct config through given id
def set_trace_config(id):
    config_list_filename = os.path.dirname(os.path.realpath(__file__)) + "/" + "trace_config_list.json"
    config_filename = os.path.dirname(os.path.realpath(__file__)) + "/" + "trace_config.json"
    with open(config_list_filename, "r", encoding="utf-8") as config_list_file:
            data = json.load(config_list_file)  # Load as a list
            selected_config = data[id]
            with open(config_filename, "w", encoding="utf-8") as f:
                json.dump(selected_config, f, indent=4)

# Get trace object
def get_LLM_trace(filename):
    trace_type, _, _, _, _ = read_trace_config("trace_config.json")
    if(trace_type == "BurstGPT"):
        return BurstGPTTrace(filename)
    elif(trace_type == "Azure"):
        return AzureLLMTrace(filename)
    return None

class Trace:
    def __init__(self, filename):
        self.trace_data = Trace.read_trace(filename)
        self.index = 0
        self.aggregated_time = 0
    
    def read_trace(filename, scale = 1):
        req_timestamps = []
        filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
        with open(filename, 'r') as trace_file:
            lines = trace_file.readlines()
            for line in lines:
                req_timestamps.append(float(line.strip('\n')) / scale)
        return req_timestamps

    def next(self):
        if(self.index >= len(self.trace_data)):
            self.aggregated_time += self.trace_data[-1]
            self.index = 0
        result = self.trace_data[self.index] + self.aggregated_time
        self.index += 1
        return result

# Abstract class
class LLMTrace(Trace):
    def __init__(self):
        self.index = 0
        self.aggregated_time = 0
        self.max_prompt_len = 3000
        self.max_gen_len = 2048
    
    def next(self):
        if(self.index >= len(self.trace_data)):
            self.aggregated_time += self.trace_data.at[self.index - 1, 'Timestamp']
            self.index = 0
        query_time = self.trace_data.at[self.index, 'Timestamp'] + self.aggregated_time
        sampled_prompt_len = self.trace_data.at[self.index, 'Request tokens']
        if sampled_prompt_len >= self.max_prompt_len:
            sampled_prompt_len = self.max_prompt_len - 1
        sampled_output_len = self.trace_data.at[self.index, 'Response tokens']
        if sampled_output_len >= self.max_gen_len:
            sampled_output_len = self.max_gen_len - 1
        self.index += 1
        return query_time, sampled_prompt_len, int(sampled_output_len)

class AzureLLMTrace(LLMTrace):
    def __init__(self, filename):
        super().__init__() 
        self.trace_data = AzureLLMTrace.read_trace(filename)
    
    def read_trace(filename):
        _, chunk_size, start_id, max_total_prompt_num, scale = read_trace_config("trace_config.json")
        filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
        chunk_iter = pd.read_csv(filename, chunksize=chunk_size)
        chunks = []
        total_prompt_num = 0
        for id, chunk in enumerate(chunk_iter):
            if(id < start_id):
                continue
            chunks.append(chunk)
            total_prompt_num += len(chunk)
            if total_prompt_num >= max_total_prompt_num:
                break
        trace = pd.concat(chunks)
        trace.rename(columns = {
                'Seconds': 'Timestamp',
                'ContextTokens': 'Request tokens',
                'GeneratedTokens': 'Response tokens'
            }, 
            inplace = True)
        trace = trace[trace['Request tokens'] != 0]
        trace = trace[trace['Response tokens'] != 0]
        trace.reset_index(drop=True, inplace=True)
        start_time = trace.at[0, 'Timestamp']
        trace['Timestamp'] = trace['Timestamp'] - start_time
        trace['Timestamp'] = trace['Timestamp'] / scale
        time_length = trace['Timestamp'].max()

        print("total trace count:", len(trace))
        print('Trace length', time_length, 'Average', len(trace) / time_length)
        return trace


class BurstGPTTrace(LLMTrace):
    def __init__(self, filename):
        super().__init__() 
        self.trace_data = BurstGPTTrace.read_trace(filename)
    
    def read_trace(filename):
        _, chunk_size, start_id, max_total_prompt_num, scale = read_trace_config("trace_config.json")
        filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
        chunk_iter = pd.read_csv(filename, chunksize=chunk_size)
        chunks = []
        total_prompt_num = 0
        for id, chunk in enumerate(chunk_iter):
            if(id < start_id):
                continue
            chunks.append(chunk)
            total_prompt_num += len(chunk)
            if total_prompt_num >= max_total_prompt_num:
                break
        trace = pd.concat(chunks)
        trace = trace[trace['Request tokens'] != 0]
        trace = trace[trace['Response tokens'] != 0]
        trace.reset_index(drop=True, inplace=True)
        start_time = trace.at[0, 'Timestamp']
        trace['Timestamp'] = trace['Timestamp'] - start_time
        trace['Timestamp'] = trace['Timestamp'] / scale
        time_length = trace['Timestamp'].max()

        print("total trace count:", len(trace))
        print('Trace length', time_length, 'Average', len(trace) / time_length)
        return trace
