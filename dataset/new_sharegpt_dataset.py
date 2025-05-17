import os
import json
import numpy as np
from dataset.sharegpt_dataset import ShareGPTDataset

class NewShareGPTDataset(ShareGPTDataset):
    def __init__(self, dataset_filename):
        super().__init__(dataset_filename)
        print("start preparing dataset")
        self.data = self.load_data_from_path(self.dataset_filepath)
        self.prefill_idx = self.get_prefill_idx()
        print("dataset preparation finished")

    def load_data_from_path(self, file_path:str):
        with open(file_path, "r") as f:
            data_json = json.load(f)
            f.close()
        processed_data = []
        for idx, data in data_json.items():
            processed_data.append((data_json[idx]["prompt"], data_json[idx]["len_prompt"]))
        processed_data.sort(key=lambda x: x[1])
        return processed_data

    def get_prefill_idx(self):
        prefill_idx = np.ones(
            (self.max_prompt_len), dtype=np.int32) * -1
        record = np.zeros(self.max_prompt_len)
        for idx, data in enumerate(self.data):
            len_prompt = data[1]
            if len_prompt >= self.max_prompt_len:
                continue
            prefill_idx[len_prompt] = idx
            record[len_prompt] = 1
        previous_idx = -1
        for idx_ii in range(self.max_prompt_len):
            if record[idx_ii] == 0:
                continue
            original_id = prefill_idx[idx_ii]
            last_mark_idx = (idx_ii + previous_idx) // 2 if previous_idx != -1 else 0
            if(previous_idx != -1):
                for __idx_ii in range(previous_idx, last_mark_idx):
                    prefill_idx[__idx_ii] = prefill_idx[previous_idx]
            for __idx_ii in range(last_mark_idx, idx_ii):
                prefill_idx[__idx_ii] = original_id
            previous_idx = idx_ii
        for idx_ii in range(previous_idx, self.max_prompt_len):
            prefill_idx[idx_ii] = original_id
        return prefill_idx

    def get_query(self, test_config):
        sampled_prompt_len = test_config.seq_len[0]
        prompt_len = self.data[self.prefill_idx[sampled_prompt_len]][1]
        return [self.data[self.prefill_idx[sampled_prompt_len]][0], prompt_len, sampled_prompt_len]