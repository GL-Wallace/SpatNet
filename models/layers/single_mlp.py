import torch.nn as nn

class singleMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features  # 保持输出维度默认与输入一致
        
        self.fc = nn.Linear(in_features, out_features)  # 单层全连接层
        self.act = act_layer() if act_layer is not None else nn.Identity()  # 可选激活函数
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')  # 保持权重初始化方式
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.fc(x)       # 单层全连接
        x = self.act(x)      # 激活函数（可选）
        x = self.drop(x)     # dropout
        return x