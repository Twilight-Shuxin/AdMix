import argparse
import os

class ProfileConfig:
    def __init__(self, args) -> None:
        self.model_name = args.model
        self.start = args.start
        self.end = args.end
        self.batch_size = args.batch_size
        self.profile_group = args.profile_group
        self.device = args.device
        self.path = args.path

def parse_profile_options():
    parser = argparse.ArgumentParser(description = "TestMPS")
    parser.add_argument(
        "--model",
        type = str,
        required = True,
        choices = ["resnet50", "vgg19", "inception_v3", "bert"],
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        required = True,
    )
    parser.add_argument(
        "--start",
        type = int,
        default = -1,
    )
    parser.add_argument(
        "--end",
        type = int,
        default = -1,
    )
    parser.add_argument(
        "--profile_group", type = int, required = True, choices=range(-1, 7)
    )
    parser.add_argument(
        "--device", type = str, default="cuda:0"
    )
    dirname, filename = os.path.split(os.path.abspath(__file__))
    parser.add_argument(
        "--path", type = str, default=dirname+"/../data"
    )
    args = parser.parse_args()
    profile_config = ProfileConfig(args=args)
    return profile_config
