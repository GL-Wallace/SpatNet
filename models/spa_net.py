import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个基本的ResNet残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 用于调整输入通道数的shortcut，确保可以与输出的通道数匹配
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)

# 定义Inception模块
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv1x3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1x5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.adjust_output  = nn.Conv1d(in_channels*4, in_channels, kernel_size=1) 
    
    def forward(self, x):
        # 保存输入，用于残差连接
        residual = x
        # print("residual.shape: ",residual.shape)

        conv1x1_out = self.conv1x1(x)
        conv1x3_out = self.conv1x3(x)
        conv1x5_out = self.conv1x5(x)
        max_pool_out = self.max_pool(x)
        
        # 合并所有输出
        output = torch.cat([conv1x1_out, conv1x3_out, conv1x5_out, max_pool_out], dim=1)
        output = self.adjust_output(output)
        print("output.shape: ",output.shape)

        
        # 通过残差连接将输入与输出相加
        return F.relu(output + residual)

# 定义最终的模型
class SpaNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SpaNet, self).__init__()
        
        # Projection层，将输入从[32, 14, 1]转换为[32, 14, 64]
        self.projection = nn.Conv1d(in_channels, 64, kernel_size=1)  # 使用1x1卷积来进行通道映射
        
        # Residual块
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        
        # Inception块
        self.inception_block = InceptionBlock(256)
        
        # 最后的全连接层，输出batch size维度
        self.fc = nn.Linear(3584, 1)  # 256个通道，4个输出来自拼接

    def forward(self, x):
        # 输入数据形状为[32, 14, 1]，我们先做一个projection
        x = x.permute(0, 2, 1)  # 转换为[32, 1, 14]
        x = self.projection(x)  # 通过1x1卷积将通道数扩展到64
        # x = x.permute(0, 2, 1)  # 转回到[32, 14, 64]
        
        # 通过ResNet块
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 通过Inception块
        x = self.inception_block(x)
        print("inception_block.shape: ", x.shape)
        # 展平数据并通过全连接层
        x = x.view(x.size(0), -1)  # 展平操作
        print("view.shape: ", x.shape)
        x = self.fc(x)             # 通过全连接层
        
        return x

# 创建模型实例
model = SpaNet()

# 示例输入数据
input_data = torch.randn(32, 14, 1)  # Batch size为32，14个特征，1个隐藏维度
output_data = model(input_data)

print(output_data.shape)  # 输出应该是[32, 32]
