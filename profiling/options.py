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

        # Save log data to under MPSTest/args.path (default: MPSTest/data)
        # Save profile data (data collected under running mode 'PROFILE') to under MPSTest/args.profile_path (default: MPSTest/profile_data)
        dirname, filename = os.path.split(os.path.abspath(__file__))
        self.path = dirname + "/../" +args.path
        self.profile_path = dirname+"/../" + args.profile_path

        self.trace_path = get_comma_list(args.trace, ArgsType.STRING, self.model_cnt)
        # Only a list [input_seq_len, output_seq_len] can be specified for llm worker.
        self.seq_len = get_comma_list(args.seq_len, ArgsType.INT)
        self.test_time = args.test_time
        self.global_tag = args.tag

        # Default runtime mode: TEST
        self.mode = RuntimeMode(args.mode)
        self.scheduler = args.scheduler

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
        default = "",
    )
    parser.add_argument(
        "--batch_size",
        type = str,
        default = "",
    )
    parser.add_argument(
        "--start",
        type = str,
        default = "",
    )
    parser.add_argument(
        "--end",
        type = str,
        default = "",
    )
    parser.add_argument(
        "--profile_group", 
        type = str,
        default = "",
    )
    parser.add_argument(
        "--device", type = str, default = "0"
    )
    parser.add_argument(
        "--path", type = str, default = "data"
    )
    parser.add_argument(
        "--profile_path", type = str, default = "profile_data"
    )
    parser.add_argument(
        "--trace", type = str, default = ''
    )
    parser.add_argument(
        "--seq_len", type = str, default = ""
    )
    parser.add_argument(
        "--tag", type = str, default = None
    )
    parser.add_argument(
        "--test_time", type = float, default = None
    )
    parser.add_argument(
        "--mode", type = str, default = RuntimeMode.TEST.value
    )
    parser.add_argument(
        "--scheduler", type = str, default = None
    )
    args = parser.parse_args()
    profile_config = ProfileConfig(args=args)
    return profile_config
