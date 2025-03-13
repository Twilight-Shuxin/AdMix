from profiling.models.split.bert import BertModel
from profiling.models.model_worker import ModelWorker
from profiling.constants import *
import torch
import random
import numpy as np

class BertWorker(ModelWorker):
    def __init__(self, mps, pipe, barrier, log_path):
        super().__init__("bert", mps, pipe, barrier, log_path)
        self.supported_seqlen = [8, 16, 32, 64]
    
    def _inference(self, input, start, end):
        return self.model.run(*input, start, end)
    
    def get_input(self, test_config):
        batch_size = test_config.batch_size
        if(test_config.nosync):
            batch_size = 1
        seq_len = random.choice(self.supported_seqlen)
        if(test_config.seq_len != -1):
            seq_len = test_config.seq_len
        if test_config.inter_flag:
            return self.partial_input[batch_size][seq_len][test_config.start], self.random_masks[batch_size][seq_len]
        else:
            return self.random_input[batch_size][seq_len], self.random_masks[batch_size][seq_len]
    
    def common_start(self):
        self.total_module = sub_module_size_dict[self.model_name]
        self.partial_input = {}
        for batch_size in self.supported_batchsize:
            self.partial_input[batch_size] = {}
            for seq_len in self.supported_seqlen:
                input = [self.random_input[batch_size][seq_len]]
                for i in range(1, self.total_module):
                    input.append(self._inference((self.random_input[batch_size][seq_len], 
                    self.random_masks[batch_size][seq_len]), 0, i))
                self.partial_input[batch_size][seq_len] = input
    
    def init_start(self):
        self.model = BertModel().half().cuda(self.device).eval()
        self.random_input = {
            batch_size: {
                seq_len: torch.LongTensor(np.random.rand(batch_size, seq_len, 768))
                    .half()
                    .cuda(self.device)
                    for seq_len in self.supported_seqlen
            }
            for batch_size in self.supported_batchsize
        }
        self.random_masks = {
            batch_size: {
                seq_len: torch.LongTensor(np.zeros((batch_size, 1, 1, seq_len)))
                .half()
                .cuda(self.device)
                for seq_len in self.supported_seqlen
            }
            for batch_size in self.supported_batchsize
        }
