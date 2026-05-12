import torch

device = torch.device("cuda")
input_data = torch.randn(32, 10).to(device)