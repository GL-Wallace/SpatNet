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
from models.cross_attender import GroupQueryAttention
from models.layers.mlp import SimpleMLP

class SpatNet(nn.Module):
    def __init__(self, input_dim_spa, input_dim_tempo, hidden_dim, output_dim):
        super(SpatNet, self).__init__()

        # Initialize SpaNet and TempoNet
        self.spa_net = SpaNet(input_dim=input_dim_spa, hidden_dim=hidden_dim)
        self.tempo_net = TempoNet(input_dim=input_dim_tempo, hidden_dim=hidden_dim)

        # Initialize the cross-attention mechanism
        self.cross_attender = GroupQueryAttention(hidden_dim, hidden_dim)

        # Final MLP layer for output
        self.simple_mlp = SimpleMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, x_spa, x_tempo):

        spa_out = self.spa_net(x_spa)  
        tempo_out = self.tempo_net(x_tempo)  

        attn_out = self.cross_attender(q = tempo_out, k = spa_out, v = spa_out)  

        output = self.simple_mlp(attn_out)  # (batch_size, output_dim)

        return output
