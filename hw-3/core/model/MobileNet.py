import torch.nn as nn

from core.model.common_modules import Conv2DBlock


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
    def __init__(self, num_classes, image_shape,
                 conv_batch_norm=False, linear_batch_norm=False, linear_dropout=False):
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
