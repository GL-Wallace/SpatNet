import math
import torch
import torch.nn as nn
from typing import Optional


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, dropout) -> None:
        super().__init__()
        self.dim = embed_dim
        self.dropout = dropout

        self.query_proj = nn.Linear(self.dim, self.dim)
        self.key_proj = nn.Linear(self.dim, self.dim)
        self.value_proj = nn.Linear(self.dim, self.dim)
        self.att_drop = nn.Dropout(self.dropout)

        self.output_proj = nn.Linear(self.dim, self.dim)

    def forward(self, X, attention_mask: Optional[torch.Tensor] = None):
        # attention_mask shape is: (batch, seq)
        # X shape is: (batch, seq, dim)

        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        
        # @ is the same as matmul()
        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)

        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        ret = self.output_proj(output)
        return ret, att_weight