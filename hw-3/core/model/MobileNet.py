from functools import partial
from typing import List, Sequence

import torch.nn as nn

from core.model.common_modules import Conv2DBlock
from util.misc import make_divisible
from util.pretrainer import get_pretrain_model

pretrain_weights = {
    'mobile_net_v2': 'https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth',
    'mobile_net_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth',
    'mobile_net_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
}


class DepthAndPointWiseConv2DBlock(nn.Module):
    def __init__(self, in_channels, point_wise_channels, kernel_size, padding, depth_stride=(1, 1), batch_norm=True):
        super().__init__()

        self.depth_wise_conv2d = Conv2DBlock(in_channels, in_channels, kernel_size, padding,
                                             groups=in_channels, stride=depth_stride, batch_norm=batch_norm)
        self.point_wise_conv2d = Conv2DBlock(in_channels, point_wise_channels, (1, 1), 'same',
                                             batch_norm=batch_norm)

    def forward(self, x):
        return self.point_wise_conv2d(self.depth_wise_conv2d(x))


class MobileNetV1(nn.Module):
    model_name = 'mobile_net_v1'

    def __init__(self, num_classes, image_shape):
        super().__init__()

        image_width, image_height = image_shape

        self.conv_1 = Conv2DBlock(3, 32, (3, 3), 1, stride=(2, 2))

        self.depth_and_point_wise_layers = nn.Sequential(
            DepthAndPointWiseConv2DBlock(32, 64, (3, 3), 'same'),
            DepthAndPointWiseConv2DBlock(64, 128, (3, 3), 1, depth_stride=(2, 2)),
            DepthAndPointWiseConv2DBlock(128, 128, (3, 3), 'same'),
            DepthAndPointWiseConv2DBlock(128, 256, (3, 3), 1, depth_stride=(2, 2)),
            DepthAndPointWiseConv2DBlock(256, 256, (3, 3), 'same'),
            DepthAndPointWiseConv2DBlock(256, 512, (3, 3), 1, depth_stride=(2, 2)),
            *[DepthAndPointWiseConv2DBlock(512, 512, (3, 3), 'same') for _ in range(5)],
            DepthAndPointWiseConv2DBlock(512, 1024, (3, 3), 1, depth_stride=(2, 2)),
            DepthAndPointWiseConv2DBlock(1024, 1024, (3, 3), 'same')
        )

        for _ in range(5):
            image_width = image_width // 2
            image_height = image_height // 2

        self.global_average_pooling2d = nn.AvgPool2d((image_width, image_height))
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.output_linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.depth_and_point_wise_layers(x)
        x = self.global_average_pooling2d(x)
        return self.output_linear(self.flatten(self.dropout(x)))


def mobile_net_v1(num_classes, image_shape):
    return MobileNetV1(num_classes, image_shape)


class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer: nn.Module = nn.ReLU, conv_layer=nn.Conv2d,
                 bias=None, inplace=None):

        if padding is None:
            padding = (kernel_size - 1) // 2
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            activation_params = {} if inplace is None else {'inplace': inplace}
            layers.append(activation_layer(**activation_params))

        super().__init__(*layers)


# noinspection PyTypeChecker
class InvertedResidualV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        """
        init inverted residual block
        :param in_channels: k
        :param out_channels: k'
        :param stride:
        :param expand_ratio: t
        """
        super().__init__()

        main_channels = in_channels * expand_ratio  # tk
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    in_channels, main_channels,
                    kernel_size=1, stride=1, activation_layer=nn.ReLU6
                )
            )

        layers.extend(
            [
                Conv2dNormActivation(
                    main_channels, main_channels,
                    stride=stride, groups=main_channels, activation_layer=nn.ReLU6
                ),
                nn.Conv2d(main_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ]
        )

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv_layers(x)
        else:
            return self.conv_layers(x)


# noinspection PyTypeChecker
class MobileNetV2(nn.Module):
    model_name = 'mobile_net_v2'

    def __init__(self, num_classes, block=InvertedResidualV2, alpha=1.0, round_nearest=8, init_weights=True):
        """
        init mobilenet v2 body net
        :param num_classes:
        :param alpha: 通道数扩展倍率
        :param round_nearest: 通道数调整为该数的倍数，方便硬件设备计算
        """
        super().__init__()

        stage_in_channels = make_divisible(32 * alpha, divisor=round_nearest)
        stage_last_channels = make_divisible(1280 * alpha, divisor=round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            # expand_ratio, each_stage_out_channels, block_num, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        all_stage_layers = [
            Conv2dNormActivation(3, stage_in_channels, stride=2, activation_layer=nn.ReLU6)
        ]

        for expand_raio, stage_out_channels, block_num, stride in inverted_residual_setting:
            stage_out_channels = make_divisible(stage_out_channels * alpha, divisor=round_nearest)
            for i in range(block_num):
                if i != 0:
                    stride = 1
                all_stage_layers.append(
                    block(stage_in_channels, stage_out_channels, stride, expand_raio)
                )
                stage_in_channels = stage_out_channels

        all_stage_layers.append(
            Conv2dNormActivation(
                stage_in_channels, stage_last_channels, kernel_size=1, activation_layer=nn.ReLU6
            )
        )

        self.feature_extraction = nn.Sequential(*all_stage_layers)

        self.connector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2)
        )

        self.fc = nn.Linear(stage_last_channels, num_classes)

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.feature_extraction(x)
        return self.fc(self.connector(x))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


def mobile_net_v2(num_classes, alpha=1.0, round_nearest=8, init_weights=True, pretrain=False, **pretrain_kwargs):
    model = MobileNetV2(num_classes, alpha=alpha, round_nearest=round_nearest, init_weights=init_weights)
    if pretrain:
        return get_pretrain_model(model, pretrain_weights.get(model.model_name), **pretrain_kwargs)
    return model.to(pretrain_kwargs.get('device'))


class SqueezeExcitation(nn.Module):
    def __init__(self, se_in_channels, squeeze_factor=4):
        """
        :param se_in_channels: 输入se模块的通道个数
        :param squeeze_factor: 第一个fc的节点个数是se_in_channels的1/squeeze_factor
        """
        super().__init__()

        squeeze_channels = make_divisible(se_in_channels // squeeze_factor, divisor=8)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_layers = nn.Sequential(
            nn.Conv2d(se_in_channels, squeeze_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(squeeze_channels, se_in_channels, kernel_size=1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        """
        :param x: 需要使用se模块的feature map
        :return:
        """
        scale = self.adaptive_avg_pool(x)
        return x * self.linear_layers(scale)


class InvertedResidualConfigV3:
    def __init__(self,
                 in_channels,
                 kernel_size,
                 expand_main_channels,
                 out_channels,
                 use_se,
                 activation,
                 stride,
                 width_multi):
        self.in_channels = InvertedResidualConfigV3.adjust_channels(in_channels, width_multi)
        self.kernel_size = kernel_size
        self.expand_main_channels = InvertedResidualConfigV3.adjust_channels(expand_main_channels, width_multi)
        self.out_channels = InvertedResidualConfigV3.adjust_channels(out_channels, width_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride

    @staticmethod
    def adjust_channels(channels, width_multi):
        return make_divisible(channels * width_multi, divisor=8)


# noinspection PyTypeChecker
class InvertedResidualV3(nn.Module):
    def __init__(self, ir_config: InvertedResidualConfigV3, norm_layer):
        super().__init__()

        if ir_config.stride not in (1, 2):
            raise ValueError(f'expect stride in (1, 2), but get stride: {ir_config.stride}')

        self.use_shortcut = (ir_config.stride == 1 and ir_config.in_channels == ir_config.out_channels)

        layers = []
        activation_layer = nn.Hardswish if ir_config.use_hs else nn.ReLU

        if ir_config.expand_main_channels != ir_config.in_channels:
            layers.append(
                Conv2dNormActivation(
                    ir_config.in_channels, ir_config.expand_main_channels,
                    kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer
                )
            )

        layers.append(
            Conv2dNormActivation(
                ir_config.expand_main_channels, ir_config.expand_main_channels,
                kernel_size=ir_config.kernel_size, stride=ir_config.stride,
                groups=ir_config.expand_main_channels, norm_layer=norm_layer, activation_layer=activation_layer
            )
        )

        if ir_config.use_se:
            layers.append(
                SqueezeExcitation(ir_config.expand_main_channels)
            )

        layers.append(
            Conv2dNormActivation(
                ir_config.expand_main_channels, ir_config.out_channels,
                kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity
            )
        )

        self.conv_layers = nn.Sequential(*layers)

        self.out_channels = ir_config.out_channels

    def forward(self, x):
        out = self.conv_layers(x)
        if self.use_shortcut:
            return x + out
        return out


# noinspection PyTypeChecker
class MobileNetV3(nn.Module):
    model_name = 'mobile_net_v3'

    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfigV3],
                 last_channels, num_classes, block=None, norm_layer=None):
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(setting, InvertedResidualConfigV3) for setting in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidualV3

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []
        first_conv_out_channels = inverted_residual_setting[0].in_channels

        layers.append(
            Conv2dNormActivation(
                3, first_conv_out_channels,
                kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish
            )
        )

        for ir_config in inverted_residual_setting:
            layers.append(block(ir_config, norm_layer=norm_layer))

        last_conv_in_channels = inverted_residual_setting[-1].out_channels
        last_conv_out_channels = 6 * last_conv_in_channels

        layers.append(
            Conv2dNormActivation(
                last_conv_in_channels, last_conv_out_channels,
                kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish
            )
        )

        self.feature_extraction = nn.Sequential(*layers)

        self.connector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(last_conv_out_channels, last_channels),
            nn.Hardswish(),
            nn.Dropout(0.2)
        )

        self.fc = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.connector(x)
        return self.fc(x)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                nn.init.zeros_(module.bias)


def mobile_net_v3_large(num_classes, reduce_tail=False, pretrain=False, **pretrain_kwargs):
    """

    :param num_classes:
    :param reduce_tail: 对于最后三层卷积层，可以设置为True进一步减少模型参数
    :param pretrain:
    :param pretrain_kwargs:
    :return:
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfigV3, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfigV3.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduce_tail else 1

    inverted_residual_setting = [
        # in_channels, kernel_size, main_channels, out_channels, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    model = MobileNetV3(inverted_residual_setting, last_channel, num_classes)
    model.model_name = 'mobile_net_v3_large'

    if pretrain:
        return get_pretrain_model(model, pretrain_weights.get(model.model_name), **pretrain_kwargs)
    return model.to(pretrain_kwargs.get('device'))


def mobile_net_v3_small(num_classes, reduce_tail, pretrain=False, **pretrain_kwargs):
    """

    :param num_classes:
    :param reduce_tail: 对于最后三层卷积层，可以设置为True进一步减少模型参数
    :param pretrain:
    :param pretrain_kwargs:
    :return:
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfigV3, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfigV3.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduce_tail else 1

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    model = MobileNetV3(inverted_residual_setting, last_channel, num_classes)
    model.model_name = 'mobile_net_v3_small'

    if pretrain:
        return get_pretrain_model(model, pretrain_weights.get(model.model_name), **pretrain_kwargs)
    return model.to(pretrain_kwargs.get('device'))

