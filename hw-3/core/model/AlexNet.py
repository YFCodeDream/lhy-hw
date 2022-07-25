import torch.nn as nn

from core.model.common_modules import Conv2DBlock, LinearBlock
from util.misc import calc_conv2d_image_shape


class AlexNet(nn.Module):
    _mod_shape_conv_layers_params = [
        ((11, 11), (1, 2), (4, 4)),
        ((3, 3), (0, 0), (2, 2)),
        ((5, 5), (2, 2), (1, 1)),
        ((3, 3), (0, 0), (2, 2)),
        ((3, 3), (0, 0), (2, 2))
    ]

    def __init__(self, num_classes, image_shape,
                 conv_batch_norm=False, linear_batch_norm=False, linear_dropout=False):
        super().__init__()

        image_width, image_height = image_shape

        self.conv2d_layers = nn.Sequential(
            Conv2DBlock(3, 96, (11, 11), (1, 2), stride=(4, 4), batch_norm=conv_batch_norm),
            nn.MaxPool2d((3, 3), stride=2),
            Conv2DBlock(96, 256, (5, 5), (2, 2), batch_norm=conv_batch_norm),
            nn.MaxPool2d((3, 3), stride=2),
            Conv2DBlock(256, 384, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(384, 384, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(384, 256, (3, 3), 'same', batch_norm=conv_batch_norm),
            nn.MaxPool2d((3, 3), stride=2)
        )

        self.flatten = nn.Flatten()

        for mod_shape_conv_layers_param in AlexNet._mod_shape_conv_layers_params:
            image_width, image_height = \
                calc_conv2d_image_shape(image_width, image_height, *mod_shape_conv_layers_param)

        self.linear_layers = nn.Sequential(
            LinearBlock(256 * image_width * image_height, 4096,
                        batch_norm=linear_batch_norm, dropout=linear_dropout),
            LinearBlock(4096, 4096, batch_norm=linear_batch_norm, dropout=linear_dropout),
            LinearBlock(4096, num_classes, batch_norm=False, dropout=False)
        )

    def forward(self, x):
        x = self.conv2d_layers(x)
        return self.linear_layers(self.flatten(x))
