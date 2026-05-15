import math

import torch
import triton
import triton.language as tl

@triton.jit
def _create_diagonal_matrix(size, values):
    rows = tl.arange(0, size)[:, None]
    cols = tl.arange(0, size)[None, :]

    diagonal_mat = tl.where(rows == cols, values[:, None], 0.0)
    return diagonal_mat


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE, ), value=float("-inf"), dtype=tl.float32)

    mask = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    q_start_idx = query_tile_index * Q_TILE_SIZE
    q_indices = tl.arange(0, Q_TILE_SIZE)[:, None] + q_start_idx
    
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        k_start_idx = j*K_TILE_SIZE
        k_indices = tl.arange(0, K_TILE_SIZE)[None, :] + k_start_idx
        if is_causal:
            mask = tl.where(q_indices >= k_indices, 0.0, -1e6)

        pre_soft_scores = tl.dot(Q_i, tl.trans(K_j))*scale + mask
        row_max = tl.max(pre_soft_scores, axis=1)
        m_i_old = m_i
        m_i = tl.maximum(m_i_old, row_max)
        correction = tl.exp(m_i_old - m_i)

        unnormalized_softmax = tl.exp(pre_soft_scores - m_i[:, None])
        l_i = correction*l_i + unnormalized_softmax.sum(axis=1)

        correction_diag = _create_diagonal_matrix(Q_TILE_SIZE, correction)
        O_i = tl.dot(correction_diag, O_i)
        O_i = tl.dot(unnormalized_softmax.to(V_j.dtype), V_j, acc=O_i)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    softmax_norm_inv = _create_diagonal_matrix(Q_TILE_SIZE, 1.0/l_i)
    O_i = tl.dot(softmax_norm_inv, O_i).to(O_block_ptr.type.element_ty)
    l_i = m_i + tl.log(l_i)
    
    tl.store(L_block_ptr, l_i, boundary_check=(0,))
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))


class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        num_batch, num_queries, d = Q.shape
        num_keys = K.shape[-2]

        ctx.Q_TILE_SIZE = 32
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal

        L = torch.zeros(num_batch, num_queries, device=Q.device)
        O = torch.zeros_like(Q)

        grid = (triton.cdiv(num_queries, ctx.Q_TILE_SIZE), num_batch)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            num_queries, num_keys,
            scale=1/math.sqrt(d), D=d,
            is_causal=ctx.is_causal,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, 
            K_TILE_SIZE=ctx.K_TILE_SIZE
        )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError