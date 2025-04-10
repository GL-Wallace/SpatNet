# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size

        # Update gate parameters
        self.w_z = nn.Linear(input_size, hidden_size)
        self.u_z = nn.Linear(input_size, hidden_size)
        # Reset gate parameters
        self.w_r = nn.Linear(input_size, hidden_size)
        self.u_r = nn.Linear(input_size, hidden_size)
        # Candidate hidden state parameters
        self.w_h = nn.Linear(input_size, hidden_size)
        self.u_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_prev):
        z_t = torch.sigmoid(self.w_z(x) + self.u_z(h_prev))
        r_t = torch.sigmoid(self.w_r(x) + self.u_r(h_prev))
        h_hat_t = torch.tanh(self.w_h(x) + self.u_h(r_t * h_prev))
        h_t = (1 - z_t) * h_prev + z_t * h_hat_t
        return h_t

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_t = torch.zeros(x.size(1), self.hidden_size)
        for t in range(x.size(0)):
            h_t = self.gru_cell(x[t], h_t)

        out = self.fc(h_t)
        return out

class BiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Forward GRU parameters
        self.forward_gru = GRUCell(input_size, hidden_size)
        
        # Backward GRU parameters
        self.backward_gru = GRUCell(input_size, hidden_size)
    
    def forward(self, x, h_prev_forward, h_prev_backward):
        # Forward pass
        h_forward = self.forward_gru(x, h_prev_forward)
        
        # Backward pass (we process the sequence in reverse)
        h_backward = self.backward_gru(x, h_prev_backward)
        
        return h_forward, h_backward

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.bigru_cell = BiGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for concatenated forward and backward states
    
    def forward(self, x):
        batch_size = x.size(1)
        # Initialize hidden states for forward and backward passes
        h_forward = torch.zeros(batch_size, self.hidden_size)
        h_backward = torch.zeros(batch_size, self.hidden_size)
        
        # Lists to store hidden states at each time step
        forward_states = []
        backward_states = []
        
        # Forward pass
        for t in range(x.size(0)):
            h_forward, _ = self.bigru_cell(x[t], h_forward, None)
            forward_states.append(h_forward)
        
        # Backward pass
        for t in reversed(range(x.size(0))):
            _, h_backward = self.bigru_cell(x[t], None, h_backward)
            backward_states.insert(0, h_backward)  # Insert at beginning to maintain order
        
        # Concatenate last forward and backward states
        h_concat = torch.cat((forward_states[-1], backward_states), dim=1)
        
        out = self.fc(h_concat)
        return out

class TempoNet(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers, dropout):
        super(TempoNet, self).__init__()
        self.num_levels = num_layers
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = BiGRU(input_dim if i == 0 else out_dim, 
                         out_dim, 
                         out_dim)
            self.levels.append(level)
    
    def forward(self, x):
        b, y, m, c = x.size()
        x = x.reshape(b, y, m*c)
        out = []
        for level in self.levels:
            x = level(x)
            out.append(x)
        return out

