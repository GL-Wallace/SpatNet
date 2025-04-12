import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.cam import ChannelAttention


class TempoNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super(TempoNet, self).__init__()
        self.num_heads = 8
        
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
            need_weights=False  # 不返回注意力权重矩阵
        )

        

        out = self.channel_attention(attn_output)

        # out = attn_output.mean(1) # []
        # print("mean:", out.shape)
        out = self.final_proj(out[:,-1,:]).squeeze(-1)
        # print("final:", out.shape, out)

        
        return out[:,]

