import os
import json
import numpy as np
from dataset.dataset import Dataset

class ShareGPTDataset(Dataset):
    def __init__(self, dataset_filename):
        self.dataset_filepath = os.path.dirname(os.path.realpath(__file__)) + "/" + dataset_filename
        self.max_prompt_len = 3000
        self.max_gen_len = 2048