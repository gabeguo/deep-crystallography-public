import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

# Credit to https://arxiv.org/pdf/2006.08195.pdf
# for the semi-periodic activation function
class Snake(nn.Module):
    def __init__(self, a=0.25):
        super(Snake, self).__init__()
        self.a = a
        return
    def forward(self, x):
        # x + 1/a * sin^2(ax)
        return x + 1.0 / self.a * torch.pow(torch.sin(self.a * x), 2)

class DeepDiffractionPatternEmbedder(nn.Module):
    def __init__(self):
        super(DeepDiffractionPatternEmbedder, self).__init__()
 
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256)
        )

        return

    """
    Input is 1000-dimensional
    Output is 256-dimensional
    """
    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1000

        x_ret = self.fc(x)

        assert x_ret.shape[1] == 256
        return x_ret

class DeepPositionEmbedder(nn.Module):
    def __init__(self):
        super(DeepPositionEmbedder, self).__init__()

        self.num_freq = 10

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
        x = torch.cat([torch.sin(2**i * np.pi * x) for i in range(0, self.num_freq)] + \
                    [torch.cos(2**i * np.pi * x) for i in range(0, self.num_freq)], 
                    dim=1)
        assert len(x.shape) == 2
        assert x.shape[1] == 60

        return x

class DeepChargeDensityRegressor(nn.Module):
    def __init__(self):
        super(DeepChargeDensityRegressor, self).__init__()

        self.diffraction_embedder = DeepDiffractionPatternEmbedder()
        self.position_embedder = DeepPositionEmbedder()
        
        self.block1 = nn.Sequential(
            nn.Linear(60 + 256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(60 + 256 + 256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        print("\toh i am so deep")

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

        assert diffraction_embedding.shape[1] == 256
        assert position_embedding.shape[1] == 60

        x0 = torch.cat((diffraction_embedding, position_embedding), dim=1)

        x1 = self.block1(x0)
        x1_combined = torch.cat((x1, x0), dim=1)
        x2 = self.block2(x1_combined)

        assert x2.shape[1] == 1
        assert len(x2.shape) == 2

        return x2.squeeze()