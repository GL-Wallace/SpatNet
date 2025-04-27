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

class SpaNet(nn.Module):
    def __init__(self, in_channels, dropout):
        super(SpaNet, self).__init__()
        self.dropout = dropout
        # 空间投影模块，包含批量归一化、ReLU激活和Dropout
        self.spa_proj = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        # 第一个卷积块，包含卷积、批量归一化和ReLU激活
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 第二个卷积块，包含卷积、批量归一化、ReLU激活和最大池化
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三个卷积块，包含卷积、批量归一化和ReLU激活
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 第四个卷积块，包含卷积、批量归一化和ReLU激活
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x_spa):
        # 通过空间投影模块
        x = self.spa_proj(x_spa)
        # 通过第一个卷积块
        x = self.conv_block1(x)
        # 通过第二个卷积块
        x = self.conv_block2(x)
        # 通过第三个卷积块
        x = self.conv_block3(x)
        # 通过第四个卷积块
        spa_out = self.conv_block4(x)

        return spa_out
# class SpaNet(nn.Module):
#     def __init__(self, in_channels, dropout):
#         super(SpaNet, self).__init__()
#         self.dropout = dropout
#         self.spa_proj = nn.Sequential(
#             nn.BatchNorm2d(num_features=in_channels),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#         )
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3,)
    
    
#     def forward(self, x_spa):
#         x = self.spa_proj(x_spa)
#         x = self.conv1(x)
#         spa_out = self.conv2(x)

#         return spa_out

class TempoNet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim=64, num_layers=2, dropout=0.1, is_submodel=False):
        super(TempoNet, self).__init__()
        self.num_heads = 8
        self.is_submodel = is_submodel

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.norm_after_gru = nn.LayerNorm(hidden_size)


    def forward(self, x, lengths=None):
        # x: [B, Y, M, C] --> flatten per-timestep features
        b, y, m, c = x.size()
        x = x.reshape(b, y, m * c)  # -> [B, Y, input_size]

        x = self.input_proj(x)      # -> [B, Y, embedding_dim]
        
        gru_out, _ = self.lstm(x)  # -> [B, Y, hidden]
        gru_out = self.norm_after_gru(gru_out)

        return gru_out

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

class MTNet(nn.Module):
    def __init__(self, in_channels, input_dim_tempo, hidden_dim, num_layers, num_heads, dropout):
        super(MTNet, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.spa_net = SpaNet(in_channels=in_channels, dropout=self.dropout)
        self.tempo_net = TempoNet(input_size=input_dim_tempo, hidden_size=hidden_dim, num_layers=num_layers, dropout=self.dropout, is_submodel=True)                
        self.fc_final = nn.Linear(hidden_dim, 1)
        self.self_attn = SelfAttention(
            embed_dim=hidden_dim,
            dropout=dropout
        )
       
    def forward(self, x_spa, x_tempo): 
        spa_out = self.spa_net(x_spa)
        B, C, H, W =  spa_out.shape
        spa_out = spa_out.permute(0, 2, 3, 1).view(B, H*W, C)
        tempo_out = self.tempo_net(x_tempo)
        feature = torch.concatenate((spa_out,tempo_out),dim=1)
        attn_out = self.self_attn(feature)
        x_pool = attn_out.mean(dim=1) 
        pred = self.fc_final(x_pool).squeeze(1)
        return pred