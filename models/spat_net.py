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
from models.spa_net import SpaNet
from models.tempo_net import TempoNet
from models.cross_attender import crossAttender
from models.layers.mlp import Mlp

class SpatNet(nn.Module):
    def __init__(self, input_dim_spa, input_dim_tempo, hidden_dim, output_dim, num_heads):
        super(SpatNet, self).__init__()
        self.num_heads = num_heads
        # Initialize SpaNet and TempoNet and CrossAttender
        self.spa_net = SpaNet(in_channels=input_dim_spa, hidden_dim=hidden_dim, is_submodel=True)
        self.tempo_net = TempoNet(input_size=input_dim_tempo, hidden_size=hidden_dim//2, num_layers=2, dropout=0.1, is_submodel=True)
        self.cross_attender = crossAttender(embed_dim=hidden_dim, num_heads=self.num_heads,nums_key_value_head = self.num_heads // 2, dropout=0.0)

        self.simple_mlp = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=output_dim, drop=0.2)
        self.fc_final = nn.Linear(3, 1)
    def forward(self, x_spa, x_tempo): 

        spa_out = self.spa_net(x_spa)  
        tempo_out = self.tempo_net(x_tempo)  
        print(f"spa_out.shapde:{spa_out.shape}, tempo_out.shape:{tempo_out.shape};\n")
        attn_out = self.cross_attender(Q = tempo_out, K = spa_out, V = spa_out)  
        print(f"attn_out.shapde:{attn_out.shape}\n")
        output = self.simple_mlp(attn_out)  # (batch_size, output_dim)
        print(f"MLP output .shapde:{output.shape}\n")

        pred = self.fc_final(output.squeeze(-1)).reshape(-1)
        return pred
