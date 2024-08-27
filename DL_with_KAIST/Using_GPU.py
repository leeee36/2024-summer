import torch

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())

device = torch.device("mps")

print(device)
print(torch.backends.mps.is_available())