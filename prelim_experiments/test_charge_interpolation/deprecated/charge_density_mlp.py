import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class DiffractionPatternEmbedder(nn.Module):
    def __init__(self):
        super(DiffractionPatternEmbedder, self).__init__()

        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)

        self.activation = nn.ReLU()

        return

    """
    Input is 1000-dimensional
    Output is 128-dimensional
    """
    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1000

        x = self.activation(self.fc1(x))
        x1 = x # skip connection
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x + x1))
        x = self.activation(self.fc6(x))

        # normalize!
        # x = torch.nn.functional.normalize(x, dim=-1)
        assert x.shape[1] == 512
        return x

class PositionEmbedder(nn.Module):
    def __init__(self):
        super(PositionEmbedder, self).__init__()

        self.num_freq = 10

        self.activation = nn.ReLU()

        return
    
    def to(self, device):
        on_device_model = super().to(device)
        return on_device_model
    
    """
    Input is 3-dimensional
    Output is 100-dimensional
    # Uses Fourier Features: https://bmild.github.io/fourfeat/
    """
    def forward(self, x):
        assert tuple(x[0].shape) == (3,)

        # sin & cosine
        x = torch.cat([torch.sin(2 * np.pi * i * x) for i in range(0, self.num_freq)] + \
                    [torch.cos(2 * np.pi * i * x) for i in range(0, self.num_freq)], 
                    dim=1)
        assert len(x.shape) == 2
        assert x.shape[1] == 60

        return x

class ChargeDensityRegressor(nn.Module):
    def __init__(self):
        super(ChargeDensityRegressor, self).__init__()

        self.diffraction_embedder = DiffractionPatternEmbedder()
        self.position_embedder = PositionEmbedder()
        
        self.fc1 = nn.Linear(60 + 512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)
        self.activation = nn.ReLU()
    
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
        assert position_embedding.shape[1] == 60

        x = torch.cat((diffraction_embedding, position_embedding), dim=1)

        x = self.activation(self.fc1(x))
        x1 = x # skip connection
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x + x1))
        x = self.fc6(x) # no ReLU, so we can have negative values

        assert x.shape[1] == 1
        assert len(x.shape) == 2

        return x.squeeze()