# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.layers.SingleHeadAttention import SingleHeadAttention


class TempoNet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim=128, num_layers=2, dropout=0.1, is_submodel=False):
        super(TempoNet, self).__init__()
        self.num_heads = 8
        self.is_submodel = is_submodel

        # projection 
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BiLSTM
        self.bigru = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.norm_after_gru = nn.LayerNorm(hidden_size * 2)

        self.self_attn = SingleHeadAttention(
            embed_dim=hidden_size*2,
            dropout=dropout
        )
        self.norm_after_attn = nn.LayerNorm(hidden_size * 2)

        self.final_proj = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths=None):
        # x: [B, Y, M, C] --> flatten per-timestep features
        b, y, m, c = x.size()
        x = x.reshape(b, y, m * c)  # -> [B, Y, input_size]

        x = self.input_proj(x)      # -> [B, Y, embedding_dim]
        
        gru_out, _ = self.bigru(x)  # -> [B, Y, hidden*2]
        gru_out = self.norm_after_gru(gru_out)
        attn_out, attn_weight= self.self_attn(X=gru_out)
        # print('attn_weight: ', attn_weight.shape)
        attn_out = self.norm_after_attn(attn_out + gru_out)


        if self.is_submodel:
            return gru_out
        else:
            out = attn_out.mean(dim=1)
            out =  self.final_proj(out).squeeze(-1)
            return out, attn_weight
