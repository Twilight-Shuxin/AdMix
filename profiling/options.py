import argparse
import os
from profiling.constants import *

class ProfileConfig:
    def __init__(self, args) -> None:
        self.model_names = get_comma_list(args.model, ArgsType.STRING)
        self.model_cnt = len(self.model_names)
        self.mps_list = get_comma_list(args.mps, ArgsType.INT, self.model_cnt)
        self.starts = get_comma_list(args.start, ArgsType.INT, self.model_cnt)
        self.ends = get_comma_list(args.end, ArgsType.INT, self.model_cnt)
        self.batch_sizes = get_comma_list(args.batch_size, ArgsType.INT, self.model_cnt)
        self.profile_groups = get_comma_list(args.profile_group, ArgsType.INT)
        self.device = args.device
        self.path = args.path
        self.trace_path = get_comma_list(args.trace, ArgsType.STRING, self.model_cnt)
        self.seq_len = get_comma_list_or_single(args.seqlen, ArgsType.INT)
        self.test_time = args.test_time
        self.global_tag = args.tag

def parse_profile_options():
    parser = argparse.ArgumentParser(description = "TestMPS")
    parser.add_argument(
        "--model",
        type = str,
        required = True,
    )
    parser.add_argument(
        "--mps",
        type = str,
        default = "-1",
    )
    parser.add_argument(
        "--batch_size",
        type = str,
        default = "1",
    )
    parser.add_argument(
        "--start",
        type = str,
        default = "-1",
    )
    parser.add_argument(
        "--end",
        type = str,
        default = "-1",
    )
    parser.add_argument(
        "--profile_group", type = str, required = True
    )
    parser.add_argument(
        "--device", type = int, default = 0
    )
    dirname, filename = os.path.split(os.path.abspath(__file__))
    parser.add_argument(
        "--path", type = str, default = dirname+"/../data"
    )
    parser.add_argument(
        "--trace", type = str, default = ''
    )
    parser.add_argument(
        "--seqlen", type = str, default = "-1"
    )
    parser.add_argument(
        "--tag", type = str, default = None
    )
    parser.add_argument(
        "--test_time", type = float, default = None
    )
    args = parser.parse_args()
    profile_config = ProfileConfig(args=args)
    return profile_config
