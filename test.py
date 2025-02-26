from profiling.profiler import ProfileServer
from profiling.options import parse_profile_options
import torch.multiprocessing as mp

if __name__=="__main__":
    mp.set_start_method('spawn')
    profile_config = parse_profile_options()
    profile_server = ProfileServer(profile_config)
    profile_server.run()