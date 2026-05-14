import argparse
import numpy as np
import torch
import timeit

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def benchmark_model(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    context_length: int,
    num_warmup: int,
    benchamrk_mode: str
):
    device = torch.device("cuda")
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    )
    model.to(device)

    optimizer = AdamW(model.parameters())
    dataset = np.arange(context_length*10, dtype=np.int64)

    runtimes = []
    for i in range(num_warmup+10):
        x, y = get_batch(dataset, batch_size=4, context_length=context_length, device=device.type)

        t1 = timeit.default_timer()
        preds = model(x)
        torch.cuda.synchronize()
        t2 = timeit.default_timer()

        loss = cross_entropy(preds, y)
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        t3 = timeit.default_timer()

        optimizer.step()
        torch.cuda.synchronize()
        t4 = timeit.default_timer()

        if i >= num_warmup:
            if benchamrk_mode == "fd":
                runtimes.append(t2-t1)
            elif benchamrk_mode == "fd+bd":
                runtimes.append(t3-t1)
            elif benchamrk_mode == "full":
                runtimes.append(t4-t1)
    return runtimes



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments for benchmarking script")
    parser.add_argument(
        "--d_model", required=True, type=int, help="Hidden dimension of transformer"
    )
    parser.add_argument(
        "--num_layers", required=True, type=int, help="Number of layers in the model"
    )
    parser.add_argument("--num_heads", required=True, type=int, help="Number of heads in each transformer block")
    parser.add_argument("--d_ff", type=int, required=True, help="Dimension of feed forwards layer")
    parser.add_argument("--num_warmup", type=int, required=True, help="Number of warmup steps")
    parser.add_argument("--benchmark_mode", required=True, type=str, choices=["fd", "fd+bd", "full"])
    parser.add_argument("--context_len", type=int, default=512, help="Context length of the transformer")

    args = parser.parse_args()

    runtime = benchmark_model(
        d_model=args.d_model, 
        d_ff=args.d_ff, 
        num_layers=args.num_layers, 
        num_heads=args.num_heads,
        context_length=args.context_len,
        num_warmup=args.num_warmup,
        benchamrk_mode=args.benchmark_mode
    )
    mean_time = np.array(runtime).mean()
    std = np.array(runtime).std()
    
    print("=====================================\n")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    print(f"Mean Time: {mean_time:.3f}, standard deviation: {std:.4f}\n")
    print("=====================================")