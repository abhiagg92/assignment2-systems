import math
import torch

from einops import einsum

QUERY_TILE_SIZE = 16
KEY_TILE_SIZE = 16


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        num_batch, num_queries, d = Q.shape
        num_keys = K.shape[-2]
        num_q_tiles = num_queries // QUERY_TILE_SIZE
        num_k_tiles = num_keys // KEY_TILE_SIZE
        ctx.is_causal = is_causal

        L = torch.zeros(num_batch, num_queries, device=Q.device)
        O = torch.zeros_like(Q)
        for i in range(num_q_tiles):
            Q_i = Q[:, i*QUERY_TILE_SIZE: (i+1)*QUERY_TILE_SIZE]
            O_i = torch.zeros(num_batch, QUERY_TILE_SIZE, d, device=Q.device)
            l_i = torch.zeros(num_batch, QUERY_TILE_SIZE, device=Q.device)
            m_i = -torch.inf * torch.ones(num_batch, QUERY_TILE_SIZE, device=Q.device)

            for j in range(num_k_tiles):
                K_j = K[:, j*KEY_TILE_SIZE: (j+1)*KEY_TILE_SIZE]
                V_j = V[:, j*KEY_TILE_SIZE: (j+1)*KEY_TILE_SIZE]

                mask = torch.zeros(num_batch, QUERY_TILE_SIZE, KEY_TILE_SIZE, device=Q.device)
                if is_causal:
                    mask = -1e6*torch.triu(torch.ones_like(mask), diagonal=1)
                
                pre_soft_scores = einsum(Q_i, K_j, "b tq d, b tk d -> b tq tk")/math.sqrt(d) + mask
                row_max = torch.amax(pre_soft_scores, -1)
                m_i_old = m_i.clone()
                m_i = torch.maximum(m_i_old, row_max)
                
                unnormalized_softmax = torch.exp(pre_soft_scores - m_i[..., None])
                l_i = torch.exp(m_i_old - m_i)*l_i + unnormalized_softmax.sum(-1)

                O_i = einsum(torch.diag_embed(torch.exp(m_i_old - m_i)), O_i, "b tq tq, b tq d -> b tq d") + \
                        einsum(unnormalized_softmax, V_j, "b tq tk, b tk d -> b tq d")
        
            O_i = einsum(torch.inverse(torch.diag_embed(l_i)), O_i, "b tq tq, b tq d -> b tq d")
            L_i = m_i + torch.log(l_i)
            L[:, i*QUERY_TILE_SIZE: (i+1)*QUERY_TILE_SIZE] = L_i
            O[:, i*QUERY_TILE_SIZE: (i+1)*QUERY_TILE_SIZE] = O_i
        
        ctx.save_for_backward(Q, K, V, L, O)
        return O

    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, L, O = ctx.saved_tensors
        batch, num_q, d_model = Q.shape
        num_k = K.shape[1]
        is_causal = ctx.is_causal
        
        D = (O*grad_output).sum(-1)
        mask = torch.zeros(batch, num_q, num_k, device=Q.device)
        if is_causal:
            mask = -1e6*torch.triu(torch.ones_like(mask), diagonal=1)
        
        scale = 1/math.sqrt(d_model)
        S = einsum(Q, K, "b tq d, b tk d -> b tq tk")*scale + mask

        P = torch.exp(S - L[..., None])

        dV = einsum(P, grad_output, "b tq tk, b tq d -> b tk d")
        dP = einsum(grad_output, V, "b tq d, b tk d -> b tq tk")
        dS = P*(dP - D[..., None])
        dQ = einsum(dS, K, "b tq tk, b tk d -> b tq d")*scale
        dK = einsum(dS, Q, "b tq tk, b tq d -> b tk d")*scale

        return dQ, dK, dV, None
