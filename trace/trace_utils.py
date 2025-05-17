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

def read_burstgpt_trace(filename, scale = None, max_total_prompt_num=15000):
    foo = os.environ.get('GVLLM_TRACE_SCALE')
    if(foo is not None):
        scale = float(foo)
    else:
        if(scale == None):
            scale = 51
    filename = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
    chunk_iter = pd.read_csv(filename, chunksize=100)
    chunks = []
    total_prompt_num = 0
    for id, chunk in enumerate(chunk_iter):
        if(id <= 10):
            continue
        chunks.append(chunk)
        total_prompt_num += len(chunk)
        if total_prompt_num >= max_total_prompt_num:
            break
    print("total trace count:", total_prompt_num)
    trace = pd.concat(chunks)
    trace = trace[trace['Request tokens'] != 0]
    trace = trace[trace['Response tokens'] != 0]
    trace.reset_index(drop=True, inplace=True)
    start_time = trace.at[0, 'Timestamp']
    trace['Timestamp'] = trace['Timestamp'] - start_time
    print("max request tokens: ", trace['Request tokens'].max(), "; min request tokens",  trace['Request tokens'].min())
    print("max response tokens: ", trace['Response tokens'].max(), "; min response tokens", trace['Response tokens'].min())
    trace['Timestamp'] = trace['Timestamp'] / scale
    time_length = trace['Timestamp'].max()
    print('Trace length', time_length, 'Average', total_prompt_num / time_length)
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
