import torch

print(f"cuda is available: {torch.cuda.is_available()}\n")
print(f"torch.cuda.get_device_name {torch.cuda.get_device_name()}")