import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim,hidden_dim,dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.norm = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(32, hidden_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # x = self.fc3(x)
        # x = self.relu2(x)
        # x = self.dropout3(x)
        # x = self.fc4(x)
        return x