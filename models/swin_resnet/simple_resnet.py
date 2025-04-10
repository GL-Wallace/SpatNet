import torch
import torch.nn as nn


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InceptionResidualBlock, self).__init__()

        self.branch11 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch33 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch55 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_combine = nn.Sequential(
            nn.Conv2d(4 * out_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch11(x)
        b2 = self.branch33(x)
        b3 = self.branch55(x)
        b4 = self.branch_pool(x)
        out = torch.cat([b1, b2, b3, b4], 1)
        out = self.conv_combine(out)
        out += self.shortcut(x)
        return self.relu(out)


class BottleNeck(nn.Module):
    dilation = 2

    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        shortcut = self.expansion(shortcut)

        return self.relu(x + shortcut)


class SimpleResNet(nn.Module):
    def __init__(self, residual, num_classes):
        super(SimpleResNet, self).__init__()

        self.num_residuals = 2
        self.outputs = [16, 32, 64]

        self.conv1 = nn.Conv2d(3, self.outputs[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.outputs[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = residual(self.outputs[0], self.outputs[1])
        self.conv3 = residual(self.outputs[1], self.outputs[2])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv3(self.conv2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def simpleResNet(num_classes=5):
    return SimpleResNet(InceptionResidualBlock, num_classes=num_classes)
