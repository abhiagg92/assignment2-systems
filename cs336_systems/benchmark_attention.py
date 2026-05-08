import timeit
import torch
import numpy as np
from pathlib import Path

from cs336_basics.model import CausalMultiHeadSelfAttention
from cs336_basics.nn_utils import cross_entropy


def benchmark_attention(d_model: int, context_length: int, benchmark_mode='fd+bd'):
    device = torch.device('cuda')
    # rope = RotaryEmbedding(context_length, d_model)
    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=1,
        positional_encoder=None
    )
    attention.to(device)
    attention = torch.compile(attention)
    input = torch.randn(4, context_length, d_model, device=device)
    output = torch.randint(10, (4, context_length), device=device)

    runtimes = []
    num_warmup = 5
    for i in range(num_warmup+10):
        if i == num_warmup:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        t1 = timeit.default_timer()
        preds = attention(input)
        torch.cuda.synchronize()
        t2 = timeit.default_timer()

        loss = cross_entropy(preds, output)
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        t3 = timeit.default_timer()

        if i == num_warmup:
            torch.cuda.memory._dump_snapshot(f'memory_logs/memory-d_{d_model}-seq_{context_length}.pickle')
            torch.cuda.memory._record_memory_history(enabled=None)

        if i >= num_warmup:
            if benchmark_mode == "fd":
                runtimes.append(t2-t1)
            elif benchmark_mode == "fd+bd":
                runtimes.append(t3-t1)
        
    
    return runtimes


if __name__ == "__main__":
    Path("memory_logs").mkdir(parents=True, exist_ok=True)
    for s in [256, 1024, 4096, 8192]:
        for d in [16, 32, 64, 128]:
            runtimes = benchmark_attention(d, s)
            mean_time = np.mean(runtimes)
            print(f"Context Length: {s}, d_model: {d}, Mean Time: {mean_time:.3f}\n")
