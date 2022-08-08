import torch
import torch.nn as nn

from util.misc import calc_conv2d_image_shape


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **conv2d_kwargs):
        super().__init__()
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **conv2d_kwargs),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv2d_block(x)


class Inception(nn.Module):
    def __init__(self, in_channels, point_wise_out_channels,
                 branches_mid_channels, branches_out_channels, pool_out_channels):
        super().__init__()
        branch_mid_channels_1, branch_mid_channels_2 = branches_mid_channels
        branch_out_channels_1, branch_out_channels_2 = branches_out_channels

        self.conv_branch_1 = Conv2dBlock(in_channels, point_wise_out_channels, kernel_size=(1, 1))

        self.conv_branch_2 = nn.Sequential(
            Conv2dBlock(in_channels, branch_mid_channels_1, kernel_size=(1, 1)),
            Conv2dBlock(branch_mid_channels_1, branch_out_channels_1, kernel_size=(3, 3), padding='same')
        )

        self.conv_branch_3 = nn.Sequential(
            Conv2dBlock(in_channels, branch_mid_channels_2, kernel_size=(1, 1)),
            Conv2dBlock(branch_mid_channels_2, branch_out_channels_2, kernel_size=(5, 5), padding='same')
        )

        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dBlock(in_channels, pool_out_channels, kernel_size=(1, 1))
        )

        self.branches = [self.conv_branch_1, self.conv_branch_2, self.conv_branch_3, self.pool_branch]

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class InceptionAux(nn.Module):
    _inception_aux_mod_shape_params = [
        ((5, 5), (0, 0), (3, 3))
    ]

    def __init__(self, image_shape, in_channels, num_classes):
        super().__init__()

        image_width, image_height = image_shape

        self.avg_pool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = Conv2dBlock(in_channels, 128, kernel_size=(1, 1))

        for params in InceptionAux._inception_aux_mod_shape_params:
            image_width, image_height = calc_conv2d_image_shape(image_width, image_height, *params)

        print(image_width, image_height)

        self.flatten = nn.Flatten()

        self.linear_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * image_width * image_height, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv(self.avg_pool(x))
        x = self.flatten(x)
        return self.linear_layers(x)


class GoogLeNet(nn.Module):
    def __init__(self, image_shape, num_classes, aux_logits=True, init_weights=False):
        super().__init__()

        image_width, image_height = image_shape
        self.aux_logits = aux_logits

        self.front_conv_pool_layers = nn.Sequential(
            Conv2dBlock(3, 64, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2)),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Conv2dBlock(64, 64, kernel_size=(1, 1)),
            Conv2dBlock(64, 192, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        front_conv_pool_layers_mod_shape_params = [
            ((7, 7), (3, 3), (2, 2)),
            ((3, 3), (0, 0), (2, 2), True),
            ((3, 3), (0, 0), (2, 2), True)
        ]

        for params in front_conv_pool_layers_mod_shape_params:
            image_width, image_height = calc_conv2d_image_shape(image_width, image_height, *params)

        self.inception_block_1 = nn.Sequential(
            Inception(192, 64, (96, 16), (128, 32), 32),
            Inception(256, 128, (128, 32), (192, 96), 64),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Inception(480, 192, (96, 16), (208, 48), 64)
        )

        image_width, image_height = calc_conv2d_image_shape(image_width, image_height,
                                                            (3, 3), (0, 0), (2, 2), ceil_mode=True)

        if self.aux_logits:
            self.aux_1 = InceptionAux((image_width, image_height), 512, num_classes)

        self.inception_block_2 = nn.Sequential(
            Inception(512, 160, (112, 24), (224, 64), 64),
            Inception(512, 128, (128, 24), (256, 64), 64),
            Inception(512, 112, (114, 32), (288, 64), 64)
        )

        if self.aux_logits:
            self.aux_2 = InceptionAux((image_width, image_height), 528, num_classes)

        self.inception_block_3 = nn.Sequential(
            Inception(528, 256, (160, 32), (320, 128), 128),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            Inception(832, 256, (160, 32), (320, 128), 128),
            Inception(832, 384, (192, 48), (384, 128), 128)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        aux_1_res, aux_2_res = None, None
        x = self.front_conv_pool_layers(x)

        x = self.inception_block_1(x)
        if self.training and self.aux_logits:
            aux_1_res = self.aux_1(x)

        x = self.inception_block_2(x)
        if self.training and self.aux_logits:
            aux_2_res = self.aux_2(x)

        x = self.inception_block_3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux_1_res, aux_2_res

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
