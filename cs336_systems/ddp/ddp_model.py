import torch
import torch.nn as nn
import torch.distributed as dist


class DDPModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.module = base_model
        self.handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.make_hook(self.handles))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
    
    def finish_gradient_synchronization_sync(self):
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
    
    @staticmethod
    def make_hook(handles):
        def hook(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            handles.append(handle)
        return hook
        