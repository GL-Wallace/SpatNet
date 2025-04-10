import torch
import torch.nn as nn

class SpatNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpatNet, self).__init__()

