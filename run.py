import torch.multiprocessing as mp 
from admix.profiling.profiler import ProfileServer  
from admix.profiling.constants import RuntimeMode
from admix.profiling.options import parse_profile_options 
        
if __name__=="__main__":
    mp.set_start_method('spawn')
    profile_config = parse_profile_options()
    profile_server = ProfileServer(profile_config)
    profile_server.setup()
    if(profile_server.mode == RuntimeMode.PROFILE):
        profile_server.profile()
    else:
        profile_server.test()
        result, passed, all_submitted = profile_server.collect_data()
        with open('result.log', 'w') as f:
            f.write(str(result) + ' ' + str(passed) + ' ' + str(all_submitted))
    profile_server.shutdown()
