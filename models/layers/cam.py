import torch
import torch.nn as nn


# Step 1: 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.fc(x.mean(dim=1))  # 在 time_length 上进行平均池化得到每个通道的注意力
        attention_weights = attention_weights.unsqueeze(1)  # 添加一个额外的维度 [batch_size, 1, hidden_size]
        return x * attention_weights  # 加权输入