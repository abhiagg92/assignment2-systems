import torch
import torch.nn as nn
import torch.distributed as dist


class DDPModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.module = base_model

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

    def forward(self, x):
        return self.module(x)
    
    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
