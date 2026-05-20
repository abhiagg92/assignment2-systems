import torch
import torch.nn as nn
import torch.distributed as dist


class DDPModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.module = base_model

        model_params = []
        for param in self.module.parameters():
            model_params.append(param.data)

        flat_params = torch._utils._flatten_dense_tensors(model_params)
        dist.broadcast(flat_params, src=0, async_op=False)
        model_params = torch._utils._unflatten_dense_tensors(flat_params, model_params)
        
        for param, model_param in zip(self.module.parameters(), model_params):
            param.data = model_param

    def forward(self, x):
        return self.module(x)
    
    def finish_gradient_synchronization(self):
        grads = []
        for param in self.module.parameters():
            if param.grad is None:
                continue
            grads.append(param.grad)

        flat_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG, async_op=False)
        reduced_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)

        counter = 0
        for param in self.module.parameters():
            if param.grad is None:
                continue
            param.grad = reduced_grads[counter]
            counter += 1