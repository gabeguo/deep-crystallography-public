import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class DiffractionPatternEmbedder(nn.Module):
    def __init__(self, num_blocks=4, single_skip=True, double_skip=True):
        super(DiffractionPatternEmbedder, self).__init__()

        self.num_blocks = num_blocks
        self.single_skip = single_skip
        self.double_skip = double_skip

        self.initfc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU()
        )

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU()
        )

        self.conv_blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=8*(i+1), out_channels=8, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_features=8),
                nn.ReLU()
            ) 
            for i in range(self.num_blocks)]
        )

        self.lastfc = nn.Sequential(
            nn.Conv1d(in_channels=8*(self.num_blocks+1), out_channels=1, kernel_size=1, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 512), # no batchnorm here, because we want outputs to be scaled
            nn.ReLU()
        )

        return

    """
    Input is 1000-dimensional
    Output is 512-dimensional
    """
    def forward(self, x):
        assert len(x.shape) == 2
        x = self.initfc(x)

        x = torch.unsqueeze(x, dim=1)
        assert len(x.shape) == 3 # make it channels (N, C, L)
        assert x.shape[1] == 1
        assert x.shape[2] == 512

        x = self.first_conv(x)
        assert len(x.shape) == 3
        assert x.shape[1] == 8
        assert x.shape[2] == 512
        x_history = [x]
        for i, the_block in enumerate(self.conv_blocks):
            assert len(x_history) == i + 1 # make sure we are updating the history list
            x = the_block(torch.cat(x_history, dim=1))
            x_history.append(x) # add new result to running list
            assert len(x.shape) == 3
            assert x.shape[1] == 8
            assert x.shape[2] == 512
        assert len(x_history) == len(self.conv_blocks) + 1 # make sure we hit all the blocks
 
        x_final = self.lastfc(torch.cat(x_history, dim=1))

        assert len(x_final.shape) == 2
        assert x_final.shape[1] == 512
        return x_final

class PositionEmbedder(nn.Module):
    def __init__(self, num_freq=10):
        super(PositionEmbedder, self).__init__()

        self.num_freq = num_freq

        return
    
    def to(self, device):
        on_device_model = super().to(device)
        return on_device_model
    
    """
    Input is 3-dimensional
    Output is 60-dimensional
    # Uses Fourier Features: https://bmild.github.io/fourfeat/
    """
    def forward(self, x):
        assert tuple(x[0].shape) == (3,)

        # sin & cosine
        x = torch.cat([torch.sin(2 * i * np.pi * x) for i in range(1, self.num_freq + 1)] + \
                    [torch.cos(2 * i * np.pi * x) for i in range(1, self.num_freq + 1)], 
                    dim=1)
        assert len(x.shape) == 2
        assert x.shape[1] == 2 * 3 * self.num_freq

        return x

class DenseChargeDensityRegressor(nn.Module):
    def __init__(self, num_blocks=4, single_skip=True, double_skip=True, num_freq=10):
        super(DenseChargeDensityRegressor, self).__init__()

        self.single_skip = single_skip
        self.double_skip = double_skip
        self.num_blocks = num_blocks
        self.num_freq = num_freq

        self.diffraction_embedder = DiffractionPatternEmbedder(\
            num_blocks = self.num_blocks, single_skip = self.single_skip, double_skip = self.double_skip)
        self.position_embedder = PositionEmbedder(num_freq=self.num_freq)
        
        self.initfc = nn.Sequential(
            nn.Linear(6 * self.num_freq + 512, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )

        self.fc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU()
            ) for i in range(self.num_blocks)
        ])

        self.lastfc = nn.Linear(512, 1) # no normalization at the end

        print("oh i am so DENSE")

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
        assert position_embedding.shape[1] == 6 * self.num_freq

        x = torch.cat((diffraction_embedding, position_embedding), dim=1)

        x = self.initfc(x) # correct dimension
        assert len(x.shape) == 2
        assert x.shape[1] == 512
        x_history = [x]
        for i, the_block in enumerate(self.fc_blocks):
            assert len(x_history) == i + 1 # make sure we are updating it
            x = the_block(torch.sum(torch.stack(x_history, dim=0), dim=0, keepdim=False))
            x_history.append(x) # add to running list
        assert len(x_history) == len(self.fc_blocks) + 1 # make sure we have all the layer outputs
        
        x_final = self.lastfc(torch.sum(torch.stack(x_history, dim=0), dim=0, keepdim=False))

        assert x_final.shape[1] == 1
        assert len(x_final.shape) == 2

        return x_final.squeeze()