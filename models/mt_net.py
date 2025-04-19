# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.spa_net import SpaNet
from models.tempo_net import TempoNet
from models.cross_attender import CrossAttender
from models.layers.mlp import MLP
from models.layers.single_mlp import singleMLP



class MTNet(nn.Module):
    def __init__(self, input_dim_spa, input_dim_tempo, hidden_dim, output_dim, num_heads, dropout):
        super(MTNet, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        # Initialize SpaNet and TempoNet and CrossAttender
        self.spa_net = MLP(input_dim=17, hidden_dim=hidden_dim, dropout=dropout)
        self.tempo_net = TempoNet(input_size=input_dim_tempo, hidden_size=hidden_dim//2, num_layers=2, dropout=self.dropout, is_submodel=True)        
        self.simple_mlp = singleMLP(in_features=hidden_dim, out_features=hidden_dim, dropout=self.dropout)
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.self_attn = SelfAttention(
            embed_dim=hidden_dim,
            dropout=dropout
        )
    
    
    def forward(self, x_spa, x_tempo): 
        spa_out = self.spa_net(x_spa)  
        tempo_out = self.tempo_net(x_tempo)
        spa_out = spa_out.expand(-1, 3, -1)
        feature = spa_out + tempo_out
        attn_out = self.self_attn(feature)
        # output = self.simple_mlp(attn_out)
        x_pool = attn_out.mean(dim=1) 
        pred = self.fc_final(x_pool).squeeze(1)
        return pred


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout) -> None:
        super().__init__()
        self.dim = embed_dim
        self.dropout = dropout

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.att_drop = nn.Dropout(self.dropout)
        self.output_proj = nn.Linear(self.dim, self.dim)

    def forward(self, X, attention_mask: Optional[torch.Tensor] = None):
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
        return ret