from profiling.options import parse_profile_options
import torch.multiprocessing as mp  
from profiling.profiler import ProfileServer  
from profiling.dummy import Dummy
from trace.trace_utils import BurstGPTTrace
from plotting.trace_plot import *
import time
from profiling.constants import *

# if __name__=="__main__":
#     mp.set_start_method('spawn')
#     profile_config = parse_profile_options()
#     dummy = Dummy(profile_config)
#     dummy.run()
    
# if __name__=="__main__":
#     trace = BurstGPTTrace("BurstGPT_1.csv")
    # time_len = 40
    # test_start_time = time.time()
    # time_limit = test_start_time + time_len
    # timestamps = []
    # while(True):
    #     req_time, sample_in_length, sample_out_length = trace.next()
    #     if ((req_time + test_start_time) > time_limit):
    #         break
    #     # timestamps.append((req_time, req_time + 5))
    #     timestamps.append(req_time)
    # trace_plot(timestamps, './data/trace2.png')
        
if __name__=="__main__":
    mp.set_start_method('spawn')
    profile_config = parse_profile_options()
    profile_server = ProfileServer(profile_config)
    profile_server.setup()
    profile_server.test()
    profile_server.collect_data()
    profile_server.shutdown()
