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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)
    

class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv1x3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1x5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.adjust_output  = nn.Conv1d(in_channels*4, in_channels, kernel_size=1) 
    
    def forward(self, x):
        residual = x
        # print("residual.shape: ",residual.shape)

        conv1x1_out = self.conv1x1(x)
        conv1x3_out = self.conv1x3(x)
        conv1x5_out = self.conv1x5(x)
        max_pool_out = self.max_pool(x)
        
        output = torch.cat([conv1x1_out, conv1x3_out, conv1x5_out, max_pool_out], dim=1)
        output = self.adjust_output(output)

        return F.relu(output + residual)

class SpaNet(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, is_submodel=False):
        super(SpaNet, self).__init__()
        self.is_submodel = is_submodel
        
        self.projection = nn.Linear(in_channels, 64)
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128)
        
        self.inception_block = InceptionBlock(128)
        
        self.fc = nn.Linear(128, 1) 

    def forward(self, x):
        print(x.shape)
        x = self.projection(x)
        B, L, C = x.size()
        x = x.view(B, C, L)
        print('before resNet: ', x.shape)
        x = self.res_block1(x)
        x = self.res_block2(x)
        print('after resNet: ', x.shape)
        x = self.inception_block(x)
        print("inception_block.shape: ", x.shape)

        if self.is_submodel:
            x = x.permute(0, 2, 1)
            print("SpaNet.shape: ", x.shape)
        else:
                x = x.view(x.size(0), -1)  
                print("view.shape: ", x.shape)
                x = self.fc(x)    

        return x

