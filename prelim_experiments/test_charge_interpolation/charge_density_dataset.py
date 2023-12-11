import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ChargeDensityDataset(Dataset):
    """
    Reads all charge density data from a path
    """
    def __init__(self, charge_data_dir, num_x_samples=10, num_y_samples=10, num_z_samples=10, max_num_crystals=1e6,\
                 n_bins=10, train=True, uniform_charge_sample_prob=0.5):
        self.train = train

        self.NUM_X_SAMPLES = num_x_samples
        self.NUM_Y_SAMPLES = num_y_samples
        self.NUM_Z_SAMPLES = num_z_samples

        self.MAX_NUM_CRYSTALS = max_num_crystals

        self.XRD_VECTOR_DIM = 1000

        # Only necessary when train == True
        # number of bins to split charge density into
        self.n_bins = n_bins
        # probability that we sample our point uniformly from the range of possible charge densities (0->1),
        # rather than from the possible positions
        self.uniform_charge_sample_prob = uniform_charge_sample_prob
        self.charge_densities_by_bins = list()

        self.charge_data_dir = charge_data_dir
        self.filepaths = list()
        self.charge_densities = list()
        self.sampled_charge_vectors = list()
        self.map_sizes = list()

        for root, dirs, files in os.walk(self.charge_data_dir, topdown=False):
            for name in files:
                if '.npy' not in name:
                    continue
                # store filepath
                curr_filepath = os.path.join(root, name)
                self.filepaths.append(curr_filepath)
                # store corresponding charge density map
                curr_charge_density = np.load(curr_filepath)
                # # normalize between 0 and 1
                # curr_charge_density = (curr_charge_density - curr_charge_density.min()) \
                #                     / (curr_charge_density.max() - curr_charge_density.min())
                self.charge_densities.append(curr_charge_density)
                # store dimension of the charge density map
                curr_map_size = curr_charge_density.shape
                self.map_sizes.append(curr_map_size)
                # put charge densities into bins
                self.put_charge_densities_into_bins(curr_charge_density)

                # store flattened vector version!
                flattened = torch.flatten(torch.from_numpy(curr_charge_density))
                sampled_charge_vector = [flattened[int(i / self.XRD_VECTOR_DIM * flattened.size(dim=0))] \
                        for i in range(self.XRD_VECTOR_DIM)]
                self.sampled_charge_vectors.append(torch.tensor(sampled_charge_vector).to(dtype=torch.float32))

                if len(self.charge_densities) > self.MAX_NUM_CRYSTALS:
                    break
            if len(self.charge_densities) > self.MAX_NUM_CRYSTALS:
                break
        
        assert len(self.filepaths) == len(self.charge_densities)
        assert len(self.filepaths) == len(self.map_sizes)
        assert len(self.filepaths) == len(self.sampled_charge_vectors)

        #print('\nfilepaths', ', '.join([x.split('/')[-1] for x in self.filepaths]))

        return

    """
    -> self.charge_densities_by_bins[i][j] = the list of (x, y, z) indices
        that have (standardized) charge density in range [j / n_bins, (j + 1) / n_bins] for molecule i
    -> indices i of self.charge_densities_by_bins are same as self.charge_densities and self.map_sizes
    Post:
    -> processes curr_charge_density to be split into bins
    """
    def put_charge_densities_into_bins(self, curr_charge_density):
        min_val = curr_charge_density.min()
        max_val = curr_charge_density.max()

        binned_charge_density = [[] for j in range(self.n_bins)]
        for x in range(curr_charge_density.shape[0]):
            for y in range(curr_charge_density.shape[1]):
                for z in range(curr_charge_density.shape[2]):
                    point_density = curr_charge_density[x, y, z]
                    standardized_point_density = (point_density - min_val) / (max_val - min_val)
                    # we know density is between 0 and 1 - convert to which bin
                    bin_index = min(self.n_bins - 1, int(standardized_point_density * self.n_bins))
                    # put it in the bin
                    binned_charge_density[bin_index].append((x, y, z))
        # add this binned charge density to the list
        self.charge_densities_by_bins.append(binned_charge_density)
        return

    def __len__(self):
        return len(self.filepaths) * self.NUM_X_SAMPLES * self.NUM_Y_SAMPLES * self.NUM_Z_SAMPLES

    """
    Input:
    -> index = (# x) * (# y) * (# z) * (crystal #) + (sample # from crystal)
        We load (# x) * (# y) * (# z) samples per crystal, at evenly spaced intervals,
        with order in x, y, z
    Returns:
    -> 1000-dim vector of charge densities for a given crystal
    -> (x, y, z) coordinates
    -> charge density at (x, y, z) for this crystal
    """
    def __getitem__(self, idx):
        # find the crystal and the sample number
        num_samples_per_crystal = self.NUM_X_SAMPLES * self.NUM_Y_SAMPLES * self.NUM_Z_SAMPLES
        crystal_idx = idx // num_samples_per_crystal

        # get the charge density map and its dimensions
        curr_density_map = self.charge_densities[crystal_idx]
        curr_dims = self.map_sizes[crystal_idx]
        # get the 1000-dim charge vector
        sampled_charge_vector = self.sampled_charge_vectors[crystal_idx]

        if self.train \
        and np.random.random() < self.uniform_charge_sample_prob: # intelligent sampling (CDF of samples proportional to charge density)
            curr_bin = np.random.randint(0, self.n_bins)
            while len(self.charge_densities_by_bins[crystal_idx][curr_bin]) < 1:
                curr_bin = np.random.randint(0, self.n_bins) # keep trying until we get a non-empty bin
            possible_coords = self.charge_densities_by_bins[crystal_idx][curr_bin]
            curr_coords = possible_coords[np.random.randint(0, len(possible_coords))]
            curr_charge = curr_density_map[curr_coords[0], curr_coords[1], curr_coords[2]]

            proportional_x = curr_coords[0] / curr_dims[0]
            proportional_y = curr_coords[1] / curr_dims[1]
            proportional_z = curr_coords[2] / curr_dims[2]

        else: # sample evenly
            sample_idx = idx % num_samples_per_crystal
            # find the specific x, y, z (out of # samples along each dimension)
            z = sample_idx % self.NUM_Z_SAMPLES
            y = ((sample_idx - z) // self.NUM_Z_SAMPLES) % self.NUM_Y_SAMPLES
            x = (sample_idx - z - (y * self.NUM_Z_SAMPLES)) \
                // (self.NUM_Y_SAMPLES * self.NUM_Z_SAMPLES)
            # get the proportional x, y, z (0 -> 1); randomly drawn from a bin of size 1 / self.NUM_SAMPLES
            proportional_x = np.random.uniform(low = x / self.NUM_X_SAMPLES, high = min(1, (x + 1) / self.NUM_X_SAMPLES)) \
                            if self.train else x / self.NUM_X_SAMPLES 
            proportional_y = np.random.uniform(low = y / self.NUM_Y_SAMPLES, high = min(1, (y + 1) / self.NUM_Y_SAMPLES)) \
                            if self.train else y / self.NUM_Y_SAMPLES 
            proportional_z = np.random.uniform(low = z / self.NUM_Z_SAMPLES, high = min(1, (z + 1) / self.NUM_Z_SAMPLES)) \
                            if self.train else z / self.NUM_Z_SAMPLES 
            # get the scaled x, y, z (grid coords)
            scaled_x = int(proportional_x * curr_dims[0])
            scaled_y = int(proportional_y * curr_dims[1])
            scaled_z = int(proportional_z * curr_dims[2]) 

            # un_random_proportional_x = x / self.NUM_X_SAMPLES
            # un_random_proportional_y = y / self.NUM_Y_SAMPLES
            # un_random_proportional_z = z / self.NUM_Z_SAMPLES
            # print('random proportional: {}, un-random proportional: {}'.format(proportional_x, un_random_proportional_x))

            # get the current charge
            curr_charge = curr_density_map[scaled_x, scaled_y, scaled_z]

        return sampled_charge_vector, \
                torch.tensor([proportional_x, proportional_y, proportional_z]).to(dtype=torch.float32), \
                torch.tensor(curr_charge).to(dtype=torch.float32)