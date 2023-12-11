import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class WideDiffractionPatternEmbedder(nn.Module):
    def __init__(self):
        super(WideDiffractionPatternEmbedder, self).__init__()

        self.initfc = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1000, 1024)),
            nn.GELU()
        )
        
        self.block1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
        )

        self.block3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
        )

        self.lastfc = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU()
        )

        return

    """
    Input is 1000-dimensional
    Output is 128-dimensional
    """
    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1000

        x0 = self.initfc(x) # correct dimension
        x1 = self.block1(x0)
        x2 = self.block2(x0 + x1)
        x3 = self.block3(x1 + x2)
        x4 = self.lastfc(x2 + x3)

        assert x4.shape[1] == 1024
        return x4

class WidePositionEmbedder(nn.Module):
    def __init__(self):
        super(WidePositionEmbedder, self).__init__()

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

class WideChargeDensityRegressor(nn.Module):
    def __init__(self):
        super(WideChargeDensityRegressor, self).__init__()

        self.diffraction_embedder = WideDiffractionPatternEmbedder()
        self.position_embedder = WidePositionEmbedder()
        
        self.initfc = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(90 + 1024, 1024)),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
        )

        self.block3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.GELU(),
        )

        self.lastfc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        print("oh i am so wide")

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

        assert diffraction_embedding.shape[1] == 1024
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