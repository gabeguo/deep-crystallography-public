import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class XRDEncoder(nn.Module):
    def __init__(self, num_channels=2):
        super(XRDEncoder, self).__init__()

        self.num_channels = num_channels

        layer_list = [
            nn.Sequential(
                nn.Conv1d(in_channels=self.num_channels, out_channels=64, kernel_size=3, padding=1, bias=False),
                # 1024 * 64
                nn.BatchNorm1d(num_features=64),
                nn.ReLU()
            )
        ]

        curr_num_channels = 64
        for size in [512, 256, 128]:
            # downsize to half the previous size
            layer_list.append(nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
            for i in range(2):
                layer_list.append(nn.Sequential(
                    nn.Conv1d(in_channels=curr_num_channels, out_channels=curr_num_channels//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(num_features=curr_num_channels//2),
                    nn.ReLU()
                ))
                curr_num_channels = curr_num_channels // 2

        self.layers = nn.Sequential(*layer_list)

        return

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] == self.num_channels
        assert x.shape[2] == 1024
        result = self.layers(x)
        assert result.shape[0] == batch_size
        assert result.shape[1] == 1
        assert result.shape[2] == 128

        return result

class XRDDecoder(nn.Module):
    def __init__(self, num_channels=2):
        super(XRDDecoder, self).__init__()

        self.num_channels = num_channels

        layer_list = list()

        curr_num_channels = 1

        for size in [128, 256, 512]:
            for i in range(2):
                layer_list.append(nn.Sequential(
                    nn.ConvTranspose1d(in_channels=curr_num_channels, out_channels=2*curr_num_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(num_features=2*curr_num_channels),
                    nn.ReLU()
                ))
                curr_num_channels *= 2
            layer_list.append(nn.Upsample(scale_factor=2, mode='linear'))
        
        layer_list.append(nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_features=2),
            nn.ReLU()
        ))

        self.layers = nn.Sequential(*layer_list)
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] == 1
        assert x.shape[2] == 128

        result = self.layers(x)

        assert result.shape[0] == batch_size
        assert result.shape[1] == self.num_channels
        assert result.shape[2] == 1024

        return result

class XRDAutoencoder(nn.Module):
    def __init__(self, num_channels):
        super(XRDAutoencoder, self).__init__()

        self.num_channels = num_channels

        self.encoder = XRDEncoder(self.num_channels)
        self.decoder = XRDDecoder(self.num_channels)

        return

    def forward(self, x):
        result = self.decoder(self.encoder(x))
        assert result.shape == x.shape
        return result