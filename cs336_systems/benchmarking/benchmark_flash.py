from triton.testing import do_bench
import torch
from enum import Enum

from cs336_systems.flash_attention.flash_attention_triton import FlashAttentionTritonFunc
from cs336_systems.flash_attention.flash_attention_pytorch import FlashAttentionFunc
from cs336_basics.model import scaled_dot_product_attention


class Implementation(Enum):
    TRITON = "triton"
    PYTORCH = "pytorch"
    NATIVE = "native"


dtype = torch.bfloat16
device = "cuda"
B = 1
impl = Implementation.TRITON

seq_lengths = [2**i for i in range(7, 17)]
d_models = [16, 32, 64, 128]
for N in seq_lengths:
    for D in d_models:
        q = torch.randn((B, N, D), device=device, dtype=dtype)
        k = torch.randn((B, N, D), device=device, dtype=dtype)
        v = torch.randn((B, N, D), device=device, dtype=dtype)

        def flash_fwd():
            if impl == Implementation.TRITON:
                FlashAttentionTritonFunc.apply(q, k, v)
            elif impl == Implementation.PYTORCH:
                FlashAttentionFunc.apply(q, k, v)
            elif impl == Implementation.NATIVE:
                scaled_dot_product_attention(q, k, v)
            else:
                print("Invalid model")

        flash_fwd()
        torch.cuda.synchronize()

        ms = do_bench(
            flash_fwd,
            warmup=5,          # often safer than default for Triton kernels
            rep=10,
            return_mode="median"
        )

        print(f"Seq Length: {N}, Model Dim: {D}, FlashAttention fwd: {ms:.3f} ms")