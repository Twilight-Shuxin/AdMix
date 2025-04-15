from profiling.options import parse_profile_options
import torch.multiprocessing as mp    
from profiling.dummy import Dummy

if __name__=="__main__":
    mp.set_start_method('spawn')
    profile_config = parse_profile_options()
    dummy = Dummy(profile_config)
    dummy.run()
    