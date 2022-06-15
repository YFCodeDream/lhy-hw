import torch.nn as nn

from core.model.common_modules import Conv2DBlock, LinearBlock


class VGG16Net(nn.Module):
    def __init__(self, num_classes, image_shape,
                 conv_batch_norm=False, linear_batch_norm=False, linear_dropout=False):
        super().__init__()

        image_width, image_height = image_shape
        # (B, C, W, H)

        self.conv_layers_1 = nn.Sequential(
            Conv2DBlock(3, 64, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(64, 64, (3, 3), 'same', batch_norm=conv_batch_norm),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        # (B, 64, W // 2, H // 2)

        self.conv_layers_2 = nn.Sequential(
            Conv2DBlock(64, 128, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(128, 128, (3, 3), 'same', batch_norm=conv_batch_norm),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        # (B, 128, W // 2 // 2, H // 2 // 2)

        self.conv_layers_3 = nn.Sequential(
            Conv2DBlock(128, 256, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(256, 256, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(256, 256, (3, 3), 'same', batch_norm=conv_batch_norm),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        # (B, 256, W // 2 // 2 // 2, H // 2 // 2 // 2)

        self.conv_layers_4 = nn.Sequential(
            Conv2DBlock(256, 512, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(512, 512, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(512, 512, (3, 3), 'same', batch_norm=conv_batch_norm),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        # (B, 512, W // 2 // 2 // 2 // 2, H // 2 // 2 // 2 // 2)

        self.conv_layers_5 = nn.Sequential(
            Conv2DBlock(512, 512, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(512, 512, (3, 3), 'same', batch_norm=conv_batch_norm),
            Conv2DBlock(512, 512, (3, 3), 'same', batch_norm=conv_batch_norm),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        # (B, 512, W // 2 // 2 // 2 // 2 // 2, H // 2 // 2 // 2 // 2 // 2)

        self.conv_layers = nn.Sequential(
            self.conv_layers_1,
            self.conv_layers_2,
            self.conv_layers_3,
            self.conv_layers_4,
            self.conv_layers_5
        )

        self.flatten = nn.Flatten()

        for i in range(len(self.conv_layers)):
            image_width = image_width // 2
            image_height = image_height // 2

        self.linear_connection = LinearBlock(512 * image_width * image_height, 4096,
                                             batch_norm=linear_batch_norm, dropout=linear_dropout)

        self.linear_layers = nn.Sequential(
            LinearBlock(4096, 4096, batch_norm=linear_batch_norm, dropout=linear_dropout),
            LinearBlock(4096, num_classes, batch_norm=False, dropout=False)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_connection(x)
        return self.linear_layers(x)
