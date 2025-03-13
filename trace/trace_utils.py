import os
import pandas as pd

def read_trace(filename, scale = 1):
    req_timestamps = []
    filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
    with open(filename, 'r') as trace_file:
        lines = trace_file.readlines()
        for line in lines:
            req_timestamps.append(float(line.strip('\n')) / scale)
    return req_timestamps

def read_burstgpt_trace(filename, scale = 200, total_prompt_num=10000):
    filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
    chunk_iter = pd.read_csv(filename, chunksize=20)
    chunks = []
    total_prompt_num = 0
    for chunk in chunk_iter:
        chunks.append(chunk)
        total_prompt_num += len(chunk)
        if total_prompt_num >= total_prompt_num:
            break
    trace = pd.concat(chunks)
    trace.reset_index(drop=True, inplace=True)
    trace['Timestamp'] = trace['Timestamp'] / scale
    return trace

class Trace:
    def __init__(self, filename):
        self.trace_data = read_trace(filename)
        self.index = 0
        self.aggregated_time = 0

    def next(self):
        if(self.index >= len(self.trace_data)):
            self.aggregated_time += self.trace_data[-1]
            self.index = 0
        result = self.trace_data[self.index] + self.aggregated_time
        self.index += 1
        return result

class BurstGPTTrace(Trace):
    def __init__(self, filename):
        self.trace_data = read_burstgpt_trace(filename)
        self.index = 0
        self.aggregated_time = 0
        self.max_prompt_len = 1024
        self.max_gen_len = 1024
    
    def next(self):
        if(self.index >= len(self.trace_data)):
            self.aggregated_time += self.trace_data.at[self.index -1, 'Timestamp']
            self.index = 0
        query_time = self.trace_data.at[self.index, 'Timestamp'] + self.aggregated_time
        sampled_prompt_len = self.trace_data.at[self.index, 'Request tokens']
        if sampled_prompt_len >= self.max_prompt_len:
            sampled_prompt_len = self.max_prompt_len - 1
        sampled_output_len = self.trace_data.at[self.index, 'Response tokens']
        if sampled_output_len >= self.max_gen_len:
            sampled_output_len = self.max_gen_len - 1
        self.index += 1
        return query_time, sampled_prompt_len, sampled_output_len
