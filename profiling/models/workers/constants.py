from enum import Enum

class SyncType(Enum):
    NO_SYNC = 0
    SINGLE = 1
    ALL = 2

class WorkerTestConfig():
    def __init__(self, test_config):
        self.batch_size = test_config.batch_size
        self.start = test_config.start
        self.end = test_config.end
        self.seq_len = test_config.seq_len
        self.nosync = test_config.nosync
        self.inter_flag = test_config.inter_flag
        self.trace_path = test_config.trace_path
        self.record_statistics = test_config.record_statistics
        self.record_statistics_saved = None
        self.test_time = test_config.test_time
        self.test_time_default = test_config.test_time_default
    
    def set_warmup(self):
        self.record_statistics_saved = self.record_statistics
        self.record_statistics = False
    
    def set_test(self):
        if(self.record_statistics_saved):
            self.record_statistics = self.record_statistics_saved
            self.record_statistics_saved = None
