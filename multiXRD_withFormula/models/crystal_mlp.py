import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class FormulaEmbedder(nn.Module):
    def __init__(self, num_blocks = 4):
        super(FormulaEmbedder, self).__init__()

        self.num_blocks = num_blocks

        if self.num_blocks == 0: # safeguard against accidentally calculating
            return

        self.initfc = nn.Sequential(
            nn.Linear(119, 512),
            nn.BatchNorm1d(num_features=512),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
            )
            for i in range(self.num_blocks - 1)
        ])

        return
    
    """
    Input is 119-dimensional
    Output is 512-dimensional
    """
    def forward(self, x):
        if self.num_blocks == 0:
            raise ValueError('not supposed to be using formula embedder')
        
        assert len(x.shape) == 2
        assert x.shape[1] == 119
        x = self.initfc(x)
        for layer in self.layers:
            x = layer(x)
        assert x.shape[1] == 512
        return x

class LatticeEmbedder(nn.Module):
    def __init__(self, num_blocks = 4):
        super(LatticeEmbedder, self).__init__()

        self.num_blocks = num_blocks

        if self.num_blocks == 0: # safeguard against using lattice embedder
            return

        self.initfc = nn.Sequential(
            nn.Linear(9, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU()
            )
            for i in range(self.num_blocks - 1)
        ])

        return
    
    """
    Input is 9-dimensional
    Output is 512-dimensional
    """
    def forward(self, x):
        if self.num_blocks == 0:
            raise ValueError('not supposed to be using lattice embedder')

        assert len(x.shape) == 2
        assert x.shape[1] == 9
        x = self.initfc(x)
        for layer in self.layers:
            x = layer(x)
        assert x.shape[1] == 512
        return x
    
class SpaceGroupEmbedder(nn.Module):
    def __init__(self, num_blocks = 4):
        super(SpaceGroupEmbedder, self).__init__()

        self.num_blocks = num_blocks

        if self.num_blocks == 0: # safeguard against using spacegroup embedder
            return

        self.initfc = nn.Sequential(
            nn.Linear(230+1, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU()
            )
            for i in range(self.num_blocks - 1)
        ])

        return
    
    """
    Input is 231-dimensional
    Output is 512-dimensional
    """
    def forward(self, x):
        if self.num_blocks == 0:
            raise ValueError('not supposed to be using spacegroup embedder')

        assert len(x.shape) == 2
        assert x.shape[1] == 231
        x = self.initfc(x)
        for layer in self.layers:
            x = layer(x)
        assert x.shape[1] == 512
        return x

class DiffractionPatternEmbedder(nn.Module):
    def __init__(self, num_blocks=6, num_channels=2):
        super(DiffractionPatternEmbedder, self).__init__()

        self.num_blocks = num_blocks
        self.num_channels = num_channels

        if self.num_blocks == 0 or self.num_channels == 0: # safeguard against using XRD embedder
            return

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels, out_channels=8*self.num_channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([8*self.num_channels, 1024]),
            nn.ReLU()
        )

        self.conv_blocks_1 = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=8*(i+self.num_channels), out_channels=8, kernel_size=3, padding=1, bias=False),
                nn.LayerNorm([8, 1024]),
                nn.ReLU()
            ) 
            for i in range(self.num_blocks)]
        )

        self.transition = nn.Sequential(
            nn.Conv1d(in_channels=8*(self.num_blocks+self.num_channels), out_channels=8*2*self.num_channels, kernel_size=1, bias=False),
            nn.LayerNorm([8*2*self.num_channels, 1024]),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # decrease dimesionality to 512
        )

        self.conv_blocks_2 = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=8*(i+2*self.num_channels), out_channels=8, kernel_size=3, padding=1, bias=False),
                nn.LayerNorm([8, 512]),
                nn.ReLU()
            ) 
            for i in range(self.num_blocks)]
        )

        self.lastfc = nn.Sequential(
            nn.Conv1d(in_channels=8*(self.num_blocks+2*self.num_channels), out_channels=1, kernel_size=1, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(num_features=512)
        )

        print('conv blocks: {}'.format(self.num_blocks))

        return

    """
    Input is (n_channels, 1024)-dimensional
    Output is 512-dimensional
    """
    def forward(self, x):
        if self.num_channels == 0 or self.num_blocks == 0:
            raise ValueError('not supposed to use XRD embedder')
        batch_size = x.shape[0]

        assert len(x.shape) == 3 # make it channels (N, C, L)
        assert x.shape[1] >= self.num_channels
        if self.num_channels < x.shape[1]:
            x = x[:, :self.num_channels, :] # only consider the relevant channels
        assert x.shape[2] == 1024 # size of XRD peak vector

        # first convolution: give it some channels
        x = self.first_conv(x)
        assert len(x.shape) == 3
        assert x.shape[0] == batch_size
        assert x.shape[1] == 8 * self.num_channels
        assert x.shape[2] == 1024

        # first group of densely connected conv blocks - size: (batch_size x num_channels x 1024)
        x_history_1 = [x]
        for i, the_block in enumerate(self.conv_blocks_1):
            assert len(x_history_1) == i + 1 # make sure we are updating the history list
            x = the_block(torch.cat(x_history_1, dim=1))
            x_history_1.append(x) # add new result to running list
            assert len(x.shape) == 3
            assert x.shape[1] == 8
            assert x.shape[2] == 1024
        assert len(x_history_1) == len(self.conv_blocks_1) + 1 # make sure we hit all the blocks
 
        # transition layer: downsize combo of all previous feature maps to (batch_size, (2 * num_channels), 512)
        x = self.transition(torch.cat(x_history_1, dim=1))
        assert len(x.shape) == 3
        assert x.shape[1] == 8 * 2 * self.num_channels
        assert x.shape[2] == 512

        # second group of densely connected conv blocks
        x_history_2 = [x] # start with downsized
        for i, the_block in enumerate(self.conv_blocks_2):
            assert len(x_history_2) == i + 1 # make sure we are updating the history list
            x = the_block(torch.cat(x_history_2, dim=1))
            x_history_2.append(x) # add new result to running list
            assert len(x.shape) == 3
            assert x.shape[1] == 8
            assert x.shape[2] == 512
        assert len(x_history_2) == len(self.conv_blocks_2) + 1 # make sure we hit all the blocks

        # get final output
        x_final = self.lastfc(torch.cat(x_history_2, dim=1))

        assert len(x_final.shape) == 2
        assert x_final.shape[0] == batch_size
        assert x_final.shape[1] == 512
        return x_final

class PositionEmbedder(nn.Module):
    def __init__(self, num_freq=128, sigma=1):
        super(PositionEmbedder, self).__init__()

        self.num_freq = num_freq
        self.sigma = sigma

        self.freq = nn.Linear(in_features=3, out_features=self.num_freq)
        print('non-learnable frequencies: {}'.format(self.sigma))
        with torch.no_grad(): # fix these weights
            self.freq.weight = nn.Parameter(torch.normal(mean=0, std=self.sigma, size=(self.num_freq, 3)), requires_grad=False)
            self.freq.bias = nn.Parameter(torch.zeros(self.num_freq), requires_grad=False)

        self.layers = nn.Sequential(
            nn.Linear(2*self.num_freq, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # # process spherical coordiantes
        # self.layers = nn.Sequential(
        #     nn.Linear(5, 512),
        #     nn.BatchNorm1d(num_features=512),
        #     nn.ReLU(),
        #     nn.Linear(512, 2 * self.num_freq)
        # )

        return
    
    def to(self, device):
        on_device_model = super().to(device)
        return on_device_model
    
    """
    Input is 3-dimensional
    Output is 2*self.num_freq-dimensional
    # Uses Fourier Features: https://bmild.github.io/fourfeat/
    """
    def forward(self, x):
        assert tuple(x[0].shape) == (3,)
        x = x - 0.5 # center at 0, so that we can have symmetries
        assert torch.min(x) >= -0.5 - 1e-4
        assert torch.max(x) <= +0.5 + 1e-4

        # # Spherical coordinates
        # r = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2 + x[:,2:3]**2) # radius length
        # assert r.shape == (x.shape[0], 1)
        # assert torch.max(r) <= 1 + 1e-4
        # assert torch.min(r) >= 0 - 1e-4

        # hypotenuse = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2) # hypotenuse length (cylindrical)
        # assert hypotenuse.shape == r.shape
        # assert torch.max(hypotenuse) <= 1 + 1e-4
        # assert torch.min(hypotenuse) >= 0 - 1e-4

        # base_cos = x[:,0:1] / (hypotenuse + 1e-20) # represent base angle continuously
        # base_sin = x[:,1:2] / (hypotenuse + 1e-20)
        # assert base_cos.shape == r.shape
        # assert base_sin.shape == r.shape
        # assert torch.max(base_cos) <= 1 + 1e-4
        # assert torch.min(base_cos) >= -1 - 1e-4
        # assert torch.max(base_sin) <= 1 + 1e-4
        # assert torch.min(base_sin) >= -1 - 1e-4

        # axis_cos = x[:,2:3] / (r + 1e-20) # represent angle from axis continuously
        # axis_sin = hypotenuse / (r + 1e-20)
        # assert axis_cos.shape == r.shape
        # assert axis_sin.shape == r.shape
        # assert torch.max(axis_cos) <= 1 + 1e-4
        # assert torch.min(axis_cos) >= -1 - 1e-4
        # assert torch.max(axis_sin) <= 1 + 1e-4
        # assert torch.min(axis_sin) >= 0 - 1e-4

        # spherical_coords = torch.cat([r, base_cos, base_sin, axis_cos, axis_sin], dim=1)
        # assert spherical_coords.shape == (x.shape[0], 5)

        # retval = self.layers(spherical_coords)
        # assert retval.shape == (x.shape[0], 2 * self.num_freq)

        # return retval

        # sin & cosine
        #print('random frequencies with sigma {}'.format(self.random_freq))
        x = self.freq(x)
        x = torch.cat([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], 
                    dim=1)
        assert len(x.shape) == 2
        assert x.shape[1] == 2 * self.num_freq

        x = self.layers(x)
        assert x.shape[1] == 512

        return x

"""
Takes in:
-> multi-channel XRD
-> empirical formula (ratios)
-> unit cell info (flattened lattice vectors)
-> space group info (one-hot code from 1->230)
Notes:
-> num_channels=0 means we don't consider XRD
-> num_formula_blocks=0 means we don't consider formula
"""
class ChargeDensityRegressor(nn.Module):
    def __init__(self, num_channels=2, num_conv_blocks=4, num_formula_blocks=4, num_lattice_blocks=4, num_spacegroup_blocks=4,
                 num_regressor_blocks=3, num_freq=128, sigma=3, dropout_prob=0):
        super(ChargeDensityRegressor, self).__init__()

        self.num_channels = num_channels
        self.num_conv_blocks = num_conv_blocks
        self.num_freq = num_freq
        self.sigma = sigma
        self.dropout_prob = dropout_prob
        self.num_regressor_blocks = num_regressor_blocks
        self.num_formula_blocks = num_formula_blocks
        # extra info (not supposed to be here)
        self.num_lattice_blocks = num_lattice_blocks
        self.num_spacegroup_blocks = num_spacegroup_blocks

        # deterministic position embedding
        self.position_embedder = PositionEmbedder(num_freq=self.num_freq, sigma=self.sigma)

        # variational approach to latent vector
        self.diffraction_embedder_mean = DiffractionPatternEmbedder(num_channels=self.num_channels, num_blocks=self.num_conv_blocks)
        self.diffraction_embedder_std = DiffractionPatternEmbedder(num_channels=self.num_channels, num_blocks=self.num_conv_blocks)
        self.formula_embedder_mean = FormulaEmbedder(num_blocks=self.num_formula_blocks)
        self.formula_embedder_std = FormulaEmbedder(num_blocks=self.num_formula_blocks)

        # extra info (not supposed to be here)
        self.lattice_embedder = LatticeEmbedder(num_blocks=self.num_lattice_blocks)
        self.spacegroup_embedder = SpaceGroupEmbedder(num_blocks=self.num_spacegroup_blocks)

        # TODO: adjust embedding size based on whether we have spacegroup & lattice?
        print('dropout prob: {}'.format(self.dropout_prob))
        print('regressor blocks: {}'.format(self.num_regressor_blocks))
        print('formula blocks: {}'.format(self.num_formula_blocks))
        print('lattice blocks: {}'.format(self.num_lattice_blocks))
        print('spacegroup blocks: {}'.format(self.num_spacegroup_blocks))
        # gamma(z) * x + beta(z)
        self.film_scale = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 512),#2 * self.num_freq),
            nn.BatchNorm1d(num_features=512),#2 * self.num_freq),
            nn.ReLU()
        )
        self.film_bias = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 512),#2 * self.num_freq),
            nn.BatchNorm1d(num_features=512),#2 * self.num_freq),
            nn.ReLU()
        )

        self.initfc = nn.Sequential(
            nn.Linear(512, 512),
            #nn.Linear(2 * self.num_freq, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Dropout(p=self.dropout_prob),
                nn.Linear(512, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU()
            ) for i in range(3)]
        )

        self.middle_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=self.dropout_prob),
                nn.Linear(512 + 512, 512),
                #nn.Linear(512 + 2 * self.num_freq, 512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU(), 
                *[nn.Sequential(
                    nn.Dropout(p=self.dropout_prob),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(num_features=512),
                    nn.ReLU()
                ) for i in range(3)]
            )
            for i in range(self.num_regressor_blocks - 1)
        ])

        self.lastfc = nn.Linear(512, 1)

        print('did you remember your formula sheet?')

        return
    
    # def to(self, device):
    #     on_device_regressor = super().to(device)
    #     on_device_regressor.position_embedder = self.position_embedder.to(device)
    #     on_device_regressor.diffraction_embedder_mean = self.diffraction_embedder_mean.to(device)
    #     on_device_regressor.diffraction_embedder_std = self.diffraction_embedder_std.to(device)
    #     on_device_regressor.formula_embedder_mean = self.formula_embedder_mean.to(device)
    #     on_device_regressor.formula_embedder_std = self.formula_embedder_std.to(device)
    #     on_device_regressor.lattice_embedder = self.lattice_embedder.to(device)
    #     on_device_regressor.spacegroup_embedder = self.spacegroup_embedder.to(device)
    #     return on_device_regressor
        
    def forward(self, diffraction_pattern, formula_vector, lattice_vector, spacegroup_vector, position, \
                diffraction_embedding=None, position_embedding=None, formula_embedding=None, lattice_embedding=None, spacegroup_embedding=None):
        # sanity check
        if (self.num_channels == 0 or self.num_conv_blocks == 0) and (diffraction_embedding is not None):
            raise ValueError('not supposed to use diffraction embedding here')
        if (self.num_formula_blocks == 0) and (formula_embedding is not None):
            raise ValueError('not supposed to use formula embedding here')
        if (self.num_lattice_blocks == 0) and (lattice_embedding is not None):
            raise ValueError('not supposed to use lattice embedding here')
        if (self.num_spacegroup_blocks == 0) and (spacegroup_embedding is not None):
            raise ValueError('not supposed to use spacegroup embedding here')
        
        # create embeddings for XRD, formula, lattice, spacegroup
        if diffraction_embedding is None:
            if self.num_channels > 0 and self.num_conv_blocks > 0:
                de_mean = self.diffraction_embedder_mean(diffraction_pattern)
                de_std = self.diffraction_embedder_std(diffraction_pattern)
                de_noise = torch.normal(mean=0, std=1, size=de_mean.shape).to(de_std.get_device())
                diffraction_embedding = de_mean + de_noise * de_std
            else:
                diffraction_embedding = torch.zeros(position.shape[0], 512).to(diffraction_pattern.get_device())
        if formula_embedding is None:
            if self.num_formula_blocks > 0: 
                fe_mean = self.formula_embedder_mean(formula_vector)
                fe_std = self.formula_embedder_std(formula_vector)
                fe_noise = torch.normal(mean=0, std=1, size=fe_mean.shape).to(fe_std.get_device())
                formula_embedding = fe_mean + fe_noise * fe_std
            else:
                formula_embedding = torch.zeros(position.shape[0], 512).to(formula_vector.get_device())
        if lattice_embedding is None:
            if self.num_lattice_blocks > 0: # do we include?
                lattice_embedding = self.lattice_embedder(lattice_vector)
            else:
                lattice_embedding = torch.zeros(position.shape[0], 512).to(lattice_vector.get_device())
        if spacegroup_embedding is None:
            if self.num_spacegroup_blocks > 0:
                spacegroup_embedding = self.spacegroup_embedder(spacegroup_vector)
            else:
                spacegroup_embedding = torch.zeros(position.shape[0], 512).to(spacegroup_vector.get_device())
        
        assert diffraction_embedding.shape[1] == 512
        assert formula_embedding.shape[1] == 512
        assert lattice_embedding.shape[1] == 512
        assert spacegroup_embedding.shape[1] == 512

        # create conditioning vector
        if self.num_spacegroup_blocks > 0 and self.num_lattice_blocks > 0:
            conditioning_vector = torch.cat((diffraction_embedding, formula_embedding,
                                            lattice_embedding, spacegroup_embedding), dim=1) # we really shouldn't have lattice & spacegroup in practice
        else:
            conditioning_vector = torch.cat((diffraction_embedding, formula_embedding), dim=1)
        assert conditioning_vector.shape[1] == 1024
        assert len(conditioning_vector.shape) == 2

        # encode position
        if position_embedding is None:
            position_embedding = self.position_embedder(position)
        assert position_embedding.shape[1] == 512#2 * self.num_freq

        # create film
        cond_scale = self.film_scale(conditioning_vector)
        assert cond_scale.shape[1] == 512#2 * self.num_freq
        cond_bias = self.film_bias(conditioning_vector)
        assert cond_bias.shape[1] == 512#2 * self.num_freq
        # condition coordinates on FiLM
        x_input = cond_scale * position_embedding + cond_bias
        assert x_input.shape[1] == 512#2 * self.num_freq

        # do the main processing blocks
        x = self.initfc(x_input)
        for i, the_block in enumerate(self.middle_blocks):
            assert len(x.shape) == 2
            assert x.shape[1] == 512
            x = the_block(torch.cat((x, x_input), dim=1))
        
        x_final = self.lastfc(x)

        assert x_final.shape[1] == 1
        assert len(x_final.shape) == 2

        return x_final.squeeze()