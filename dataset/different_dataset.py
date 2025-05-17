import os
import json
import numpy as np
from dataset.sharegpt_dataset import ShareGPTDataset

class DifferentDataset(ShareGPTDataset):
    def __init__(self, dataset_filename):
        super().__init__(dataset_filename)
        self.data = self.load_data_from_path(self.dataset_filepath)
        self.query_id = 0

    def load_data_from_path(self, file_path:str):
        with open(file_path, "r") as f:
            data_json = json.load(f)
            f.close()
        processed_data = []
        for idx, data in data_json.items():
            processed_data.append((data_json[idx]["prompt"], data_json[idx]["len_prompt"]))
        processed_data.sort(key=lambda x: x[1])
        return processed_data

    def get_query(self, test_config):
        if(test_config != None):
            sampled_prompt_len = test_config.seq_len[0]
            return [self.data[self.prefill_idx[sampled_prompt_len]][0], prompt_len, sampled_prompt_len]
        else:
            query_id = self.query_id
            self.query_id = (self.query_id + 1) % 300
            return [self.data[query_id][0]]