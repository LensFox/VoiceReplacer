import torch
import torch.nn as nn

class Netrowk(nn.Module):
    def __init__(self):
        super(Netrowk, self).__init__()
        # (513, 25, 1) => (513, 25, 1)
        self.conv2d_input = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1)
        # (513, 25, 1) => (513, 25, 32)
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1),
            nn.LeakyReLU())
        # (513, 25, 32) => (513, 25, 16)
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                padding=1),
            nn.LeakyReLU())
        # (513, 25, 16) => (171, 8, 16)
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=3)
        # (171, 8, 16) => (171, 8, 16)
        self.dropout_1 = nn.Dropout()
        # (171, 8, 16) => (171, 8, 64)
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=3,
                padding=1),
            nn.LeakyReLU())
        # (171, 8, 64) => (171, 8, 16)
        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=3,
                padding=1),
            nn.LeakyReLU())
        # (171, 8, 16) => (57, 2, 16)
        self.max_pool2d_2 = nn.MaxPool2d(kernel_size=3)
        # (57, 2, 16) => (57, 2, 16)
        self.dropout_2 = nn.Dropout()
        # (57, 2, 16) => (1824)
        self.flatten = torch.flatten
        # (1824) => (128)
        self.linear_1 = nn.Sequential(
           nn.Linear(
               in_features=1824,
               out_features=256),
           nn.LeakyReLU())
        # (128) => (128)
        self.dropout_3 = nn.Dropout()
        # (128) => (513)
        self.linear_output = nn.Linear(
            in_features=256,
            out_features=513)

    def forward(self, fragment):
        out = self.conv2d_input(fragment)
        out = self.conv2d_1(out)
        out = self.conv2d_2(out)
        out = self.max_pool2d_1(out)
        out = self.dropout_1(out)
        out = self.conv2d_3(out)
        out = self.conv2d_4(out)
        out = self.max_pool2d_2(out)
        out = self.dropout_2(out)
        out = self.flatten(out, start_dim = 1)
        out = self.linear_1(out)
        out = self.dropout_3(out)
        out = self.linear_output(out)

        return out