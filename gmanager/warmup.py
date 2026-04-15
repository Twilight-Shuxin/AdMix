import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda")

# model = nn.Linear(10, 5).to(device)
print("starting")
input_data = torch.randn(32, 10).to(device)
print("done")