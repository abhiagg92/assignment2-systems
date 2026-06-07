from typing import Any
from copy import deepcopy

import torch
from torch.optim import Optimizer
import torch.distributed as dist



class DDPOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_param_groups = []
        self.global_local_map = {}
        super().__init__(params, defaults=kwargs)
        self.optim = optimizer_cls(self.local_param_groups, **kwargs)
    
    def step(self, closure=None, **kwargs):
        loss = None if closure is None else closure()
        self.optim.step(closure, **kwargs)

        for group in self.param_groups:
            for idx, param in enumerate(group["params"]):
                owner = idx % self.world_size
                dist.broadcast(param.data, src=owner)

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        assigned_params = []
        local_param_group = deepcopy(param_group)
        for idx, param in enumerate(param_group["params"]):
            owner = idx % self.world_size
            if owner == self.rank:
                assigned_params.append(param)
        
        local_param_group["params"] = assigned_params
        self.local_param_groups.append(local_param_group)
