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

        L = torch.zeros(num_batch, num_queries)
        O = torch.zeros_like(Q)
        for i in range(num_q_tiles):
            Q_i = Q[:, i*QUERY_TILE_SIZE: (i+1)*QUERY_TILE_SIZE]
            O_i = torch.zeros(num_batch, QUERY_TILE_SIZE, d)
            l_i = torch.zeros(num_batch, QUERY_TILE_SIZE)
            m_i = -torch.inf * torch.ones(num_batch, QUERY_TILE_SIZE)

            for j in range(num_k_tiles):
                K_j = K[:, j*KEY_TILE_SIZE: (j+1)*KEY_TILE_SIZE]
                V_j = V[:, j*KEY_TILE_SIZE: (j+1)*KEY_TILE_SIZE]
                
                pre_soft_scores = einsum(Q_i, K_j, "b tq d, b tk d -> b tq tk")/math.sqrt(d)
                row_max = torch.amax(pre_soft_scores, -1)
                m_i_old = torch.tensor(m_i)
                m_i = torch.maximum(m_i_old, row_max)
                
                unnormalized_softmax = torch.exp(pre_soft_scores - m_i[..., None])
                l_i = torch.exp(m_i_old - m_i)*l_i + unnormalized_softmax.sum(-1)

                O_i = einsum(torch.diag_embed(torch.exp(m_i_old - m_i)), O_i, "b tq tq, b tq d -> b tq d") + \
                        einsum(unnormalized_softmax, V_j, "b tq tk, b tk d -> b tq d")
        
            O_i = einsum(torch.inverse(torch.diag_embed(l_i)), O_i, "b tq tq, b tq d -> b tq d")
            L_i = m_i + torch.log(l_i)
            L[:, i*QUERY_TILE_SIZE: (i+1)*QUERY_TILE_SIZE] = L_i
            O[:, i*QUERY_TILE_SIZE: (i+1)*QUERY_TILE_SIZE] = O_i
        
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
