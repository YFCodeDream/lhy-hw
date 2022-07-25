import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm=True, dropout=True, dropout_prob=0.25):
        super().__init__()
        # print(input_dim, output_dim)
        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.batch_norm = batch_norm
        self.dropout = dropout
        if self.batch_norm:
            self.batch_norm_1d = nn.BatchNorm1d(output_dim)
        if self.dropout:
            self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear_block(x)
        if self.batch_norm:
            x = self.batch_norm_1d(x)
        if self.dropout:
            x = self.dropout_layer(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=1, stride=(1, 1), batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm

        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            nn.ReLU()
        )

        if self.batch_norm:
            self.batch_norm_2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d_block(x)
        if self.batch_norm:
            x = self.batch_norm_2d(x)
        return x
