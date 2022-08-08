import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152')


class BasicBlock(nn.Module):
    """
    18 & 34
    """
    # 标志残差结构里卷积核个数的扩展倍数
    kernel_expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1), down_sample=None):
        # down_sample对应每个block第一层改变通道数的残差结构（虚线分支）
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.down_sample = down_sample

    def forward(self, x):
        x_shortcut = self.down_sample(x) if self.down_sample is not None else x

        out = self.conv_block_1(x)
        out = self.conv_block_2(out)

        out = out + x_shortcut
        return F.relu(out)


class BottleNeck(nn.Module):
    """
    50 & 101 & 152
    """
    # 标志残差结构里卷积核个数的扩展倍数
    kernel_expansion = 4

    def __init__(self, in_channels, out_channels, stride=(1, 1), down_sample=None):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(out_channels, BottleNeck.kernel_expansion * out_channels,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(BottleNeck.kernel_expansion * out_channels)
        )
        self.down_sample = down_sample

    def forward(self, x):
        x_shortcut = self.down_sample(x) if self.down_sample is not None else x

        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)

        out = out + x_shortcut

        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, block_num: list, num_classes, include_top=True, init_weight=True):
        # block_num传入一个列表，表示每个stage堆叠res block的个数
        # e.g. ResNet34为[3, 4, 6, 3]
        super().__init__()

        self.include_top = include_top

        assert len(block_num) == 4

        # stage_main_channels: 每个stage主分支上的channel数
        self.each_stage_main_channels = [64, 128, 256, 512]
        # stage_in_channels: 每个stage输入的channel数
        self.stage_in_channels = 64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, self.stage_in_channels,
                      kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False),
            nn.BatchNorm2d(self.stage_in_channels),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        all_stage_layers = self._make_stage(block, block_num[0], self.each_stage_main_channels[0])

        for i in range(1, len(self.each_stage_main_channels)):
            all_stage_layers.extend(
                self._make_stage(block, block_num[i], self.each_stage_main_channels[i], stride=(2, 2)))

        self.all_stage_layers = nn.Sequential(*all_stage_layers)

        if self.include_top:
            self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(self.each_stage_main_channels[-1] * block.kernel_expansion, num_classes)

        if init_weight:
            self._init_weights()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool(x)
        x = self.all_stage_layers(x)

        if self.include_top:
            x = self.adaptive_avg_pool(x)
            x = self.flatten(x)
            x = self.fc(x)

        return x

    def _make_stage(self, block, block_num, stage_main_channels, stride=(1, 1)):
        down_sample = None

        # stage_out_channels: 当前stage应该输出的channel数
        stage_out_channels = stage_main_channels * block.kernel_expansion

        if stride != (1, 1) or stage_out_channels != self.stage_in_channels:
            down_sample = nn.Sequential(
                nn.Conv2d(self.stage_in_channels, stage_out_channels,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(stage_out_channels)
            )

        stage_layers = [block(self.stage_in_channels, stage_main_channels,
                              down_sample=down_sample, stride=stride)]

        self.stage_in_channels = stage_out_channels

        for _ in range(1, block_num):
            stage_layers.append(block(self.stage_in_channels, stage_main_channels))

        return stage_layers

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


class ResNet18(nn.Module):
    def __init__(self, num_classes, include_top=True, init_weight=True):
        super().__init__()

        self.resnet_18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes,
                                include_top=include_top, init_weight=init_weight)

    def forward(self, x):
        return self.resnet_18(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes, include_top=True, init_weight=True):
        super().__init__()

        self.resnet_34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes,
                                include_top=include_top, init_weight=init_weight)

    def forward(self, x):
        return self.resnet_34(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes, include_top=True, init_weight=True):
        super().__init__()

        self.resnet_50 = ResNet(BottleNeck, [3, 4, 6, 3], num_classes,
                                include_top=include_top, init_weight=init_weight)

    def forward(self, x):
        return self.resnet_50(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes, include_top=True, init_weight=True):
        super().__init__()

        self.resnet_101 = ResNet(BottleNeck, [3, 4, 23, 3], num_classes,
                                 include_top=include_top, init_weight=init_weight)

    def forward(self, x):
        return self.resnet_101(x)


class ResNet152(nn.Module):
    def __init__(self, num_classes, include_top=True, init_weight=True):
        super().__init__()

        self.resnet_152 = ResNet(BottleNeck, [3, 8, 36, 3], num_classes,
                                 include_top=include_top, init_weight=init_weight)

    def forward(self, x):
        return self.resnet_152(x)
