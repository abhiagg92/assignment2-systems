'''
uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx 
--pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all 
--python-backtrace=cuda --capture-range=cudaProfilerApi 
-- python cs336_systems/benchmark_nsys.py --d_model 768 --d_ff 3072 
--num_layers 12 --num_heads 12 --num_warmup 5 --benchmark_mode full
'''


import argparse
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler
from einops import einsum
import math

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask = None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output


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
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
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

    for i in range(num_warmup+10):
        if i == num_warmup:
            profiler.start()
            print(f"{i}: PUSH")
        x, y = get_batch(dataset, batch_size=4, context_length=context_length, device=device.type)

        preds = model(x)
        loss = cross_entropy(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        profiler.stop()

        if i == num_warmup + 10 - 1:
            # profiler.stop()
            print(f"{i}: POP")


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

    benchmark_model(
        d_model=args.d_model, 
        d_ff=args.d_ff, 
        num_layers=args.num_layers, 
        num_heads=args.num_heads,
        context_length=args.context_len,
        num_warmup=args.num_warmup,
        benchamrk_mode=args.benchmark_mode
    )
