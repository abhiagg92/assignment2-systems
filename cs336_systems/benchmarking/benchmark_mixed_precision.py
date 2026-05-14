import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda")
model = ToyModel(20, 1).to(device)
x = torch.rand(4, 20).to(device)
gt = torch.rand(4, 1).to(device)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    y = model(x)
    y1 = model.fc1(x)
    y2 = model.ln(model.relu(model.fc1(x)))
    loss = mse_loss(y, gt)
    loss.backward()
    print(f"fc1 weight: {model.fc1.weight.data.dtype}")
    print(f"ln weight: {model.ln.weight.data.dtype}")
    print(f"ln bias: {model.ln.bias.data.dtype}")
    print(f"fc2 weight: {model.fc2.weight.data.dtype}")
    print(f"Input: {x.dtype}")
    print(f"Output: {y.dtype}")
    print(f"Fc1: {y1.dtype}")
    print(f"ln: {y2.dtype}")
    print(f"Loss: {loss.dtype}")
    print(f"fc1 grad: {model.fc1.weight.grad_dtype}")
    print(f"ln grad: {model.ln.weight.grad_dtype}")
    print(f"fc2 grad: {model.fc2.weight.grad.dtype}")