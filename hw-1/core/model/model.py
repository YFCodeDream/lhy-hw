import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_1d = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear_layer(x)
        if self.batch_norm:
            x = self.batch_norm_1d(x)
        return x


class COVID19Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_layers = nn.Sequential(
            LinearLayer(input_dim, 64, True),
            LinearLayer(64, 32, True),
            LinearLayer(32, 1, False)
        )

    def forward(self, x):
        return self.linear_layers(x).squeeze(1)
