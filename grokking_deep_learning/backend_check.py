import torch
import torchvision

# check torch backend
print("Torch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Is MPS (Apple Silicon GPU) available?", torch.backends.mps.is_available())

# check torchvision
print("Torchvision version:", torchvision.__version__)


# test with tensor allocation
# CPU tensor
x = torch.tensor([1.0])
print("Device (CPU):", x.device)

# MPS tensor (if available)
if torch.backends.mps.is_available():
    x_mps = x.to("mps")
    print("Device (MPS):", x_mps.device)




x = torch.arange(12, dtype=torch.float32)
print(x)