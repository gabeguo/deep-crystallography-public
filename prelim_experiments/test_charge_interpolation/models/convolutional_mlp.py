import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class ConvDiffractionPatternEmbedder(nn.Module):
    def __init__(self):
        super(ConvDiffractionPatternEmbedder, self).__init__()

        self.initfc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.GELU()
        )
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.GELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.GELU()
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.GELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.GELU()
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.GELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=4),
            nn.GELU()
        )

        self.lastfc = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 512),
            nn.GELU()
        )

        return

    """
    Input is 1000-dimensional
    Output is 128-dimensional
    """
    def forward(self, x):
        assert len(x.shape) == 2
        x = torch.unsqueeze(x, dim=1)
        assert len(x.shape) == 3
        assert x.shape[1] == 1
        assert x.shape[2] == 1000

        x0 = self.initfc(x) # correct dimension
        x1 = self.block1(x0)
        x2 = self.block2(x0 + x1)
        x3 = self.block3(x1 + x2)
        x4 = self.lastfc(x2 + x3)

        assert x4.shape[1] == 512
        return x4

class ConvPositionEmbedder(nn.Module):
    def __init__(self):
        super(ConvPositionEmbedder, self).__init__()

        self.num_freq = 15

        return
    
    def to(self, device):
        on_device_model = super().to(device)
        return on_device_model
    
    """
    Input is 3-dimensional
    Output is 90-dimensional
    # Uses Fourier Features: https://bmild.github.io/fourfeat/
    """
    def forward(self, x):
        assert tuple(x[0].shape) == (3,)

        # sin & cosine
        x = torch.cat([torch.sin(2 * np.pi * i * x) for i in range(0, self.num_freq)] + \
                    [torch.cos(2 * np.pi * i * x) for i in range(0, self.num_freq)], 
                    dim=1)
        assert len(x.shape) == 2
        assert x.shape[1] == 90

        return x

class ConvChargeDensityRegressor(nn.Module):
    def __init__(self):
        super(ConvChargeDensityRegressor, self).__init__()

        self.diffraction_embedder = ConvDiffractionPatternEmbedder()
        self.position_embedder = ConvPositionEmbedder()
        
        self.initfc = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(90 + 512, 512)),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.GELU(),
        )

        self.block3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.GELU(),
        )

        self.lastfc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        print("this is convoluted")

        return
    
    def to(self, device):
        on_device_regressor = super().to(device)
        on_device_regressor.diffraction_embedder = self.diffraction_embedder.to(device)
        on_device_regressor.position_embedder = self.position_embedder.to(device)
        return on_device_regressor
        
    def forward(self, diffraction_pattern, position, diffraction_embedding=None, position_embedding=None):
        if diffraction_embedding is None:
            diffraction_embedding = self.diffraction_embedder(diffraction_pattern)
        if position_embedding is None:
            position_embedding = self.position_embedder(position)

        assert diffraction_embedding.shape[1] == 512
        assert position_embedding.shape[1] == 90

        x = torch.cat((diffraction_embedding, position_embedding), dim=1)

        x0 = self.initfc(x) # correct dimension
        x1 = self.block1(x0)
        x2 = self.block2(x0 + x1)
        x3 = self.block3(x1 + x2)
        x4 = self.lastfc(x2 + x3)

        assert x4.shape[1] == 1
        assert len(x4.shape) == 2

        return x4.squeeze()