import os
import pandas as pd
import json

CODE_DIR_PATH=os.path.dirname(os.path.realpath(__file__))
LLM_TRACE_CONFIG_LIST_FILENAME="trace_config_list.json"
LLM_TRACE_CONFIG_FILENAME="trace_config.json"

"""
    A class used to hold LLM trace config data.
"""
class LLMTraceConfig():
    def __init__(self, **kwargs):
        self.trace_name = kwargs["trace_name"]
        self.trace_id = kwargs["trace_id"]
        self.chunk_size = kwargs["chunk_size"]
        self.start_id = kwargs["start_id"]
        self.max_prompt_cnt = kwargs["max_prompt_cnt"]
        self.scale = kwargs["scale"]
        self.test_time = 100
    
    """
        Get LLM trace tag.
            [trace_name&id, startid, chunksize, test_time, scale]
        FIXME: Now changed to new naming convention, but example follows old convention.
        E.g.[BurstGPT1,100,331,400000,1]
    """
    def get_trace_tag(self):
        trace_arg = [self.trace_name + str(self.trace_id), self.start_id, self.chunk_size, self.test_time, self.scale]
        trace_tag = ",".join([str(x) for x in trace_arg])
        return trace_tag

"""
    Read trace config from specified llm trace config json file.
"""
def read_LLM_trace_config():
    config_filepath = CODE_DIR_PATH + "/" + LLM_TRACE_CONFIG_FILENAME
    with open(config_filepath, 'r') as trace_config_file:
        data = json.load(trace_config_file)
    llm_trace_config = LLMTraceConfig(
            trace_name = data["trace"],
            trace_id = data["trace_id"],
            chunk_size = data["chunk_size"],
            start_id = data["start_id"],
            max_prompt_cnt = data["max_prompt_cnt"],
            scale = data["scale"]
        )
    if("time" in data):
        llm_trace_config.test_time = data["time"]
    return llm_trace_config  

"""
    Set LLM trace config file to correct config through given id.
"""
def set_LLM_trace_config(id):

    def get_trace_by_id():
        return next((item for item in data if str(item.get("id")) == id), None)
    
    config_list_filename = CODE_DIR_PATH + "/" + LLM_TRACE_CONFIG_LIST_FILENAME
    config_filename = CODE_DIR_PATH + "/" + LLM_TRACE_CONFIG_FILENAME
    with open(config_list_filename, "r", encoding="utf-8") as config_list_file:
            data = json.load(config_list_file)  # Load as a list
            selected_config = get_trace_by_id()
            with open(config_filename, "w", encoding="utf-8") as f:
                json.dump(selected_config, f, indent=4)

class Trace():
    def next():
        pass

class SimpleTimeTrace(Trace):
    def __init__(self, filename):
        self.trace_data = self.read_trace(filename)
        self.index = 0
        self.aggregated_time = 0
    
    def read_trace(self, filename, scale = 1):
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

class LLMTrace(Trace):
    def __init__(self, llm_trace_config):
        self.index = 0
        self.aggregated_time = 0
        self.max_prompt_len = 3000
        self.max_gen_len = 2048
        self.trace_config = llm_trace_config
        self.trace_data = self.read_trace()
    
    def read_trace(self):
        filename = CODE_DIR_PATH + "/" + self.trace_config.trace_name + str(self.trace_config.trace_id)  + ".csv"
        chunk_iter = pd.read_csv(filename, chunksize = self.trace_config.chunk_size)
        chunks = []
        start_time = None
        for id, chunk in enumerate(chunk_iter):
            if(id < self.trace_config.start_id):
                continue
            chunk = chunk[
                (chunk['Request tokens'] != 0) &
                (chunk['Response tokens'] != 0)
            ].copy()
            if(chunk.empty):
                continue
            if(start_time is None):
                start_time = chunk['Timestamp'].iloc[0]
            chunk['Timestamp'] = chunk['Timestamp'] - start_time
            chunk['Timestamp'] = chunk['Timestamp'] / self.trace_config.scale
            chunks.append(chunk)
            last_timestamp = chunk['Timestamp'].iloc[-1]
            if(last_timestamp > self.trace_config.test_time):
                break
       
        trace = pd.concat(chunks)
        trace.reset_index(drop=True, inplace=True)
        time_length = trace['Timestamp'].iloc[-1]
        print("Total trace prompt count:", len(trace), 'Trace time length:', time_length, 'Average intensity:', len(trace) / time_length)
        return trace
    
    def next(self):
        if(self.index >= len(self.trace_data)):
            self.aggregated_time += self.trace_data['Timestamp'].iloc[-1]
            self.index = 0
        query_time = self.trace_data.at[self.index, 'Timestamp'] + self.aggregated_time
        sampled_prompt_len = self.trace_data.at[self.index, 'Request tokens']
        sampled_output_len = self.trace_data.at[self.index, 'Response tokens']
        if sampled_prompt_len > self.max_prompt_len:
            sampled_prompt_len = self.max_prompt_len
        if sampled_output_len > self.max_gen_len:
            sampled_output_len = self.max_gen_len
        self.index += 1
        # FIXME: currently suppose model's context length is fixed to 4096; but this is actually subject to models; 
        # change so that it matches the corresponding model afterwards. (currently this is needed for sglang)
        return query_time, int(sampled_prompt_len), min(int(sampled_output_len), 4080 - int(sampled_prompt_len))
