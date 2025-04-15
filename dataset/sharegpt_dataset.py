import os
import json
import numpy as np

def load_data_from_path(file_path:str):
    with open(file_path, "r") as f:
        data_json = json.load(f)
        f.close()
    processed_data = []
    for idx, data in data_json.items():
        processed_data.append((data_json[idx]["prompt"], data_json[idx]["len_prompt"],
                        data_json[idx]["len_output"], data_json[idx]['output']))
    processed_data.sort(key=lambda x: (x[1], x[2]))
    return processed_data

class Dataset(object):
    def __init__(self, dataset_filename, qps=1.0, trace=None, conv_or_api=None, scale=1):
        dataset_filepath = os.path.dirname(os.path.realpath(__file__)) + "/" + dataset_filename
        self.data = load_data_from_path(dataset_filepath)
        self.qps = qps
        self.query_time = 0
        self.max_prompt_len = 1024
        self.max_gen_len = 1024
        self.prefill_idx = self.get_prefill_idx()
        self.zipf_param = 1.1
        self.gamma_shape = 0.25
        self.gamma_scale = 2
        self.query_id = 0
        self.gamma_shape_dict = dict()
        self.trace = trace
        self.conv_or_api = conv_or_api
        self.scale = scale

    def get_prefill_idx(self):
        prefill_idx = np.ones(
            (self.max_prompt_len, self.max_gen_len), dtype=np.int32) * -1
        record = np.zeros(self.max_prompt_len)
        print("prefill record")
        for idx, data in enumerate(self.data):
            len_prompt = data[1]
            len_output = data[2]
            if len_prompt >= self.max_prompt_len or len_output >= self.max_gen_len:
                continue
            prefill_idx[len_prompt][len_output] = idx
            record[len_prompt] = 1
        print(len(self.data))
        print("prefill generate idx in one prompt len")
        for idx_ii in range(self.max_prompt_len):
            if record[idx_ii] == 0:
                continue

            previous_idx = -1
            for idx_jj in range(self.max_gen_len):
                if prefill_idx[idx_ii][idx_jj] != -1:
                    for _idx_jj in range(idx_jj):
                        prefill_idx[idx_ii][_idx_jj] = prefill_idx[idx_ii][idx_jj]
                    previous_idx = idx_jj
                    break

            for idx_jj in range(previous_idx, self.max_gen_len):
                if prefill_idx[idx_ii][idx_jj] == -1:
                    flag = 0
                    for _idx_jj in range(idx_jj, min(self.max_gen_len, idx_jj*2-previous_idx)):
                        if prefill_idx[idx_ii][_idx_jj] != -1:
                            flag = 1
                            prefill_idx[idx_ii][idx_jj] = prefill_idx[idx_ii][_idx_jj]
                            break
                    if flag == 0:
                        prefill_idx[idx_ii][idx_jj] = prefill_idx[idx_ii][previous_idx]
                else:
                    previous_idx = idx_jj

        previous_idx = -1
        print("prefill prompt len")
        for _idx_ii in range(self.max_prompt_len):
            if record[_idx_ii] != 0:
                for __idx_ii in range(_idx_ii):
                    for idx_jj in range(self.max_gen_len):
                        prefill_idx[__idx_ii][idx_jj] = prefill_idx[_idx_ii][idx_jj]
                previous_idx = _idx_ii
                break

        print("prefill prompt len via prompt")
        for idx_ii in range(previous_idx, self.max_prompt_len):
            if record[idx_ii] == 0:
                flag = 0
                for _idx_ii in range(idx_ii, min(self.max_prompt_len, idx_ii*2-previous_idx)):
                    if record[_idx_ii] == 1:
                        flag = 1
                        for _idx_jj in range(self.max_gen_len):
                            prefill_idx[idx_ii][_idx_jj] = prefill_idx[_idx_ii][_idx_jj]
                        break
                if flag == 0:
                    for _idx_jj in range(self.max_gen_len):
                        prefill_idx[idx_ii][_idx_jj] = prefill_idx[previous_idx][_idx_jj]
            else:
                previous_idx = idx_ii

        return prefill_idx

    def get_query(self, test_config):
        sampled_prompt_len = test_config.seq_len[0]
        sampled_output_len = test_config.seq_len[1]
        prompt_len = self.data[self.prefill_idx[sampled_prompt_len]
                                    [sampled_output_len]][1]
        output_len = self.data[self.prefill_idx[sampled_prompt_len]
                                    [sampled_output_len]][2]
        return [self.data[self.prefill_idx[sampled_prompt_len][sampled_output_len]][0], prompt_len, sampled_prompt_len]