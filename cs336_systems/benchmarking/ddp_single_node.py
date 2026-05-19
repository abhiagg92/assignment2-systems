import os
from time import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, warmup, iters):
    setup(rank, world_size)
    num_param = 1024*1024*100//4
    data = torch.randn(num_param, dtype=torch.float32)
    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
    dist.barrier()

    start = time()
    for _ in range(iters):
        dist.all_reduce(data, async_op=False)
    dist.barrier()
    end = time()

    avg_s = (end - start) / iters
    print(f"Average time: {avg_s}")

if __name__ == "__main__":
    world_size = 6
    warmup = 5
    iters = 10
    mp.spawn(fn=distributed_demo, args=(world_size, warmup, iters, ), nprocs=world_size, join=True)
