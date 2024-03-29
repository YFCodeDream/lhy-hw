import torch.nn as nn
import torch.nn.functional as F

__all__ = ('ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50',
           'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152', 'resnext_50')

from util.pretrainer import get_pretrain_model

pretrain_weights = {
    'resnet_18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet_34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet_50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    'resnet_101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth',
    'resnet_152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth',
    'resnext_50': 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth'
}


# noinspection PyUnusedLocal
class BasicBlock(nn.Module):
    """
    18 & 34
    """
    # 标志残差结构里卷积核个数的扩展倍数
    kernel_expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1), down_sample=None, **kwargs):
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

    def __init__(self, in_channels, main_channels,
                 stride=(1, 1), down_sample=None, groups=1, width_per_group=64):
        super().__init__()

        main_width = int(main_channels * (width_per_group / 64.)) * groups

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, main_width,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(main_width),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(main_width, main_width,
                      kernel_size=(3, 3), stride=stride, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(main_width),
            nn.ReLU()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(main_width, BottleNeck.kernel_expansion * main_channels,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(BottleNeck.kernel_expansion * main_channels)
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
    def __init__(self, block, block_num: list, num_classes,
                 groups=1, width_per_group=64, include_top=True, init_weights=True):
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

        all_stage_layers = self._make_stage(block, block_num[0], self.each_stage_main_channels[0],
                                            groups=groups, width_per_group=width_per_group)

        for i in range(1, len(self.each_stage_main_channels)):
            all_stage_layers.extend(
                self._make_stage(block, block_num[i], self.each_stage_main_channels[i],
                                 stride=(2, 2), groups=groups, width_per_group=width_per_group))

        self.all_stage_layers = nn.Sequential(*all_stage_layers)

        if self.include_top:
            self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(self.each_stage_main_channels[-1] * block.kernel_expansion, num_classes)

        if init_weights:
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

    def _make_stage(self, block, block_num, stage_main_channels, stride=(1, 1), groups=1, width_per_group=64):
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
                              down_sample=down_sample, stride=stride, groups=groups, width_per_group=width_per_group)]

        for _ in range(1, block_num):
            stage_layers.append(block(stage_out_channels, stage_main_channels))

        self.stage_in_channels = stage_out_channels

        return stage_layers

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


def resnet(resnet_class, num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    model = resnet_class(num_classes, include_top, init_weights)
    if pretrain:
        return get_pretrain_model(model, pretrain_weights.get(model.model_name), **pretrain_kwargs)
    return model.to(pretrain_kwargs.get('device'))


class ResNet18(nn.Module):
    model_name = 'resnet_18'

    def __init__(self, num_classes, include_top=True, init_weights=True):
        super().__init__()

        self.resnet_18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes,
                                include_top=include_top, init_weights=init_weights)

    def forward(self, x):
        return self.resnet_18(x)


def resnet_18(num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    return resnet(ResNet18, num_classes, include_top, init_weights, pretrain, **pretrain_kwargs)


class ResNet34(nn.Module):
    model_name = 'resnet_34'

    def __init__(self, num_classes, include_top=True, init_weights=True):
        super().__init__()

        self.resnet_34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes,
                                include_top=include_top, init_weights=init_weights)

    def forward(self, x):
        return self.resnet_34(x)


def resnet_34(num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    return resnet(ResNet34, num_classes, include_top, init_weights, pretrain, **pretrain_kwargs)


class ResNet50(nn.Module):
    model_name = 'resnet_50'

    def __init__(self, num_classes, include_top=True, init_weights=True):
        super().__init__()

        self.resnet_50 = ResNet(BottleNeck, [3, 4, 6, 3], num_classes,
                                include_top=include_top, init_weights=init_weights)

    def forward(self, x):
        return self.resnet_50(x)


def resnet_50(num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    return resnet(ResNet50, num_classes, include_top, init_weights, pretrain, **pretrain_kwargs)


class ResNet101(nn.Module):
    model_name = 'resnet_101'

    def __init__(self, num_classes, include_top=True, init_weights=True):
        super().__init__()

        self.resnet_101 = ResNet(BottleNeck, [3, 4, 23, 3], num_classes,
                                 include_top=include_top, init_weights=init_weights)

    def forward(self, x):
        return self.resnet_101(x)


def resnet_101(num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    return resnet(ResNet101, num_classes, include_top, init_weights, pretrain, **pretrain_kwargs)


class ResNet152(nn.Module):
    model_name = 'resnet_152'

    def __init__(self, num_classes, include_top=True, init_weights=True):
        super().__init__()

        self.resnet_152 = ResNet(BottleNeck, [3, 8, 36, 3], num_classes,
                                 include_top=include_top, init_weights=init_weights)

    def forward(self, x):
        return self.resnet_152(x)


def resnet_152(num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    return resnet(ResNet152, num_classes, include_top, init_weights, pretrain, **pretrain_kwargs)


class ResNeXt50(nn.Module):
    model_name = 'resnext_50'

    def __init__(self, num_classes, groups=32, width_per_group=4, include_top=True, init_weights=True):
        super().__init__()

        self.resnext_50 = ResNet(BottleNeck, [3, 4, 6, 3], num_classes,
                                 groups, width_per_group, include_top, init_weights)

    def forward(self, x):
        return self.resnext_50(x)


def resnext_50(num_classes, include_top=True, init_weights=True, pretrain=False, **pretrain_kwargs):
    return resnet(ResNeXt50, num_classes, include_top, init_weights, pretrain, **pretrain_kwargs)
