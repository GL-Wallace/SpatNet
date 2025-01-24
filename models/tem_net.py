import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCell(nn.module):
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

class TemNet(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(TemNet, self).__init__()
        self.num_levels = 3
        self.levels = nn.ModuleList
        for i in range(self.num_levels):
            level = GRU(

            )
    def forward(self, x):
        out = []

