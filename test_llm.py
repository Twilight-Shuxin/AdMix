from profiling.options import parse_profile_options 
from profiling.profiler import ProfileServer  
from profiling.constants import RuntimeMode
from trace.trace_utils import BurstGPTTrace
import torch.multiprocessing as mp 
        
if __name__=="__main__":
    mp.set_start_method('spawn')
    profile_config = parse_profile_options()
    profile_server = ProfileServer(profile_config)
    profile_server.setup()
    if(profile_server.mode == RuntimeMode.PROFILE):
        print("profile")
        profile_server.profile()
    else:
        profile_server.test()
        result, passed, all_submitted = profile_server.collect_data()
        with open('result.log', 'w') as f:
            f.write(str(result) + ' ' + str(passed) + ' ' + str(all_submitted))
    profile_server.shutdown()
