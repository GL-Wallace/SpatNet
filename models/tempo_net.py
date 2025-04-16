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
from models.layers.cam import ChannelAttention


class TempoNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1, is_submodel=False):
        super(TempoNet, self).__init__()
        self.num_heads = 8
        self.is_submodel = is_submodel
        
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size*2, 
            num_heads=self.num_heads, 
            batch_first=True  # [batch, seq_len, input_dim]
        )
        # FPN
        self.fpn_convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=1, padding=0),  # 小卷积核
            nn.Conv1d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=2, padding=0),  # 中等卷积核
            nn.Conv1d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=3, padding=0)   # 大卷积核
        ])
        
        self.channel_attention = ChannelAttention(hidden_size*2)
        self.final_proj = nn.Linear(hidden_size*2, 1)


    def forward(self, x, lengths=None):
        b, y, m, c = x.size()
        x = x.reshape(b, y, m*c)

        # Bi-GRU
        gru_out, _ = self.bigru(x)  # [batch, seq_len, hidden_size*2]
        # gru_out = gru_out.permute(0, 2, 1)  # [batch, hidden_size*2, seq_len]
        attn_output, _ = self.self_attn(
            query=gru_out, 
            key=gru_out, 
            value=gru_out, 
            need_weights=False 
        )

        out = self.channel_attention(attn_output)
        print(f"channel_attention: {out.shape}")
        
        if self.is_submodel:
            print("Tempo Out Shape:", out.shape)
        else:
            out = self.final_proj(out[:,-1,:]).squeeze(-1)

        return out

