import math
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, nums_key_value_head, dropout=0.0):
        super(GroupQueryAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        assert num_heads % nums_key_value_head == 0, "nums_key_value_head must be devided by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.nums_key_value_head = num_heads // 2
        self.head_dim = embed_dim // num_heads
        self.ln = torch.nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, nums_key_value_head * self.head_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.W_rpe_k = nn.Linear(self.embed_dim, nums_key_value_head * self.head_dim)
        self.W_rpe_v = nn.Linear(self.embed_dim, nums_key_value_head * self.head_dim)

    def forward(self, Q, K, V, mask=None, rpe: Optional[torch.Tensor] = None):
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)
        B, L, _ = q.size()

        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.nums_key_value_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.nums_key_value_head, self.head_dim).permute(0, 2, 1, 3)

        if rpe is not None:
            rpe = self.ln(rpe)
            rpe_k = self.W_rpe_k(rpe).reshape(B, L, L, self.nums_key_value_head, -1).permute(0, 3, 1, 2, 4)
            rpe_v = self.W_rpe_v(rpe).reshape(B, L, L, self.nums_key_value_head, -1).permute(0, 3, 1, 2, 4)
            q1 = (
                q[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            k1 = (
                k[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            v1 = (
                v[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            k1 += rpe_k
            v1 += rpe_v

            q1 = q1.reshape(B, self.num_heads, L * L, -1)
            k1 = k1.reshape(B, self.nums_key_value_head, L * L, -1)  # [B, H, L*L, D]
            v1 = v1.reshape(B, self.nums_key_value_head, L * L, -1)  # [B, H, L*L, D]
            k1 = k1.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)
            v1 = v1.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)
            # Calculate attention within local windows
            attn_mat = (q1 * k1).sum(dim=-1) / math.sqrt(self.head_dim)
            attn_mat = F.softmax(attn_mat, dim=-1)
            attn_mat = self.attn_drop(attn_mat)
            # Weighted sum of values
            qv = ((attn_mat[:, :, :, None] * v1)
                  .reshape(B, self.num_heads, L, L, -1)
                  .sum(-2)
                  .permute(0, 2, 1, 3)
                  .reshape(B, L, -1))
        else:

            k = k.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)
            v = v.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)

            attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            qv = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)

        output = self.proj_drop(self.proj(qv))
        return output

