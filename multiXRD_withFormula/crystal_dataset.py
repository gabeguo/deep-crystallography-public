import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool, Manager
from chempy.util.parsing import formula_to_composition
from chempy.util.periodic import relative_atomic_masses
from skimage.metrics import structural_similarity as ssim
import random

"""Utils"""

# Thanks ChatGPT!
def invert_dict(d):
    """Inverts a dictionary by making its keys as values and values as keys"""
    inverted = dict()
    for key, value in d.items():
        if value not in inverted:
            inverted[value] = [key]
        else:
            inverted[value].append(key)
    return inverted

# Thanks https://stackoverflow.com/posts/33190472/timeline
def rotations24(polycube):
    """List all 24 rotations of the given 3d array"""
    def rotations4(polycube, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        for i in range(4):
             yield np.rot90(polycube, i, axes)

    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    yield from rotations4(polycube, (1,2))

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotations4(np.rot90(polycube, 2, axes=(0,2)), (1,2))

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotations4(np.rot90(polycube, axes=(0,2)), (0,1))
    yield from rotations4(np.rot90(polycube, -1, axes=(0,2)), (0,1))

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotations4(np.rot90(polycube, axes=(0,1)), (0,2))
    yield from rotations4(np.rot90(polycube, -1, axes=(0,1)), (0,2))

#############

"""
Holds:
-> charge density maps
-> multi-channel XRD patterns
-> empirical formulas
-> unit cell parameters
-> spacegroups
"""
class CrystalDataset(Dataset):
    """
    Reads all charge density data from a path
    """
    def __init__(self, charge_data_dir, xrd_data_dir, chem_formula_path, lattice_path, spacegroup_path, \
                num_x_samples=10, num_y_samples=10, num_z_samples=10, max_num_crystals=1e6, \
                num_channels=2, 
                ignore_elemental_ratios=False, standardized_formula=True, num_excluded_elems=0,
                use_mass_ratios=False,
                train=True):

        self.train = train

        self.NUM_X_SAMPLES = num_x_samples
        self.NUM_Y_SAMPLES = num_y_samples
        self.NUM_Z_SAMPLES = num_z_samples

        self.MAX_NUM_CRYSTALS = max_num_crystals

        self.num_channels = num_channels

        self.ignore_elemental_ratios = ignore_elemental_ratios
        self.standardized_formula = standardized_formula # only matters if we ignore elemental ratios
        self.num_excluded_elems = num_excluded_elems # only matters if we ignore elemental ratios

        self.use_mass_ratios = use_mass_ratios # whether to use mass percentage by element, or empirical formula (weight 1 each)

        self.XRD_VECTOR_DIM = 1024

        self.charge_data_dir = charge_data_dir
        self.xrd_data_dir = xrd_data_dir
        self.chem_formula_path = chem_formula_path
        self.lattice_path = lattice_path
        self.spacegroup_path = spacegroup_path

        self.filepaths = list()
        self.charge_densities = list()
        self.xrd_vectors = list()
        self.map_sizes = list()
        self.formulas = list()
        self.lattices = list()
        self.spacegroups_one_hot = list() # this one is a one-hot vector

        self.spacegroups_int = list() # this one is an int

        self.load_valid_crystals_and_formulas()
        self.load_lattices()
        self.load_spacegroups()

        valid_roots_and_files = list()
        for root, dirs, files in os.walk(self.charge_data_dir, topdown=False):
            for name in files:
                if '.npy' in name and self.get_mp_id(name) in self.valid_crystals:
                    valid_roots_and_files.append((root, name))

        valid_roots_and_files = valid_roots_and_files[:self.MAX_NUM_CRYSTALS]
        
        self.id_to_charge_density = dict()
        print('processing crystals')
        for item in tqdm(valid_roots_and_files):
            self.process_crystal(item)

        print(len(self.charge_densities))
        assert len(self.filepaths) == len(self.charge_densities)
        assert len(self.filepaths) == len(self.map_sizes)
        assert len(self.filepaths) == len(self.xrd_vectors)
        assert len(self.filepaths) == len(self.formulas)

        return
    
    """
    -> Sets self.id_to_flattened_lattices to a mapping of mp id to 9-dimensional flattened unit cell
    """
    def load_lattices(self):
        assert os.path.exists(self.lattice_path)
        with open(self.lattice_path) as fin:
            self.id_to_lattice_matrix = json.load(fin)
            self.id_to_flattened_lattices = {item[0]:torch.Tensor(item[1]).flatten() for item in self.id_to_lattice_matrix.items()}
            assert len(self.id_to_flattened_lattices) > 0
        return

    """
    Function to create a one-hot encoded list from class index
    Thanks ChatGPT!
    """
    def one_hot_encode(self, total_classes, class_index):
        assert class_index < total_classes
        return torch.Tensor([1 if i == class_index else 0 for i in range(total_classes)])

    """
    -> Sets self.id_to_spacegroup_one_hot to a mapping of mp id to (230 + 1)-dimensional one-hot spacegroup encoding
    -> Sets self.spacegroup_to_mpids to a mapping of spacegroup to list of mp_ids that are in that spacegroup
    """
    def load_spacegroups(self):
        assert os.path.exists(self.spacegroup_path)
        with open(self.spacegroup_path) as fin:
            self.id_to_spacegroup = json.load(fin)
            self.id_to_spacegroup_one_hot = {item[0]:self.one_hot_encode(231, item[1]) for item in self.id_to_spacegroup.items()}
        assert len(self.id_to_spacegroup_one_hot) > 0

        self.spacegroup_to_mpids = invert_dict(self.id_to_spacegroup)
        return

    """
    -> Sets self.valid_crystals to a set of the valid crystals we can have in this dataset, based on the
    JSON self.valid_crystals_file
    -> Sets self.id_to_formula to a mapping of mp id to n-hot encoding, where the encoding e
        is a 119-dimensional tensor with e[i] = % of atoms of element with atomic number i in the empirical formula (sum to 1)
    """
    def load_valid_crystals_and_formulas(self):
        assert os.path.exists(self.chem_formula_path)
        with open(self.chem_formula_path) as fin:
            self.id_to_formulaStr = json.load(fin)
            self.valid_crystals = {id for id in self.id_to_formulaStr}
            self.id_to_formula = {item[0]:self.get_vectorized_formula(item[1]) for item in self.id_to_formulaStr.items()}
            assert len(self.valid_crystals) > 0
        return

    """
    Based on instance variables 
        (e.g., self.ignore_elemental_ratios, self.standardized_formula, self.num_excluded_elems),
    returns 119-dim vectorized version of formulaStr
    """
    def get_vectorized_formula(self, formulaStr):
        if self.ignore_elemental_ratios:
            return self.to_composition_vector_no_ratio(formulaStr=formulaStr, 
                standardized=self.standardized_formula, numToRemove=self.num_excluded_elems)
        return self.to_n_hot(formulaStr, by_mass=self.use_mass_ratios)

    """
    -> Input: 
        -> formulaStr of form 'A2(B3C5)3'
        -> by_mass for whether to count ratios by mass percentage, or by atom count percentage (pretend all have atomic weight 1)
    -> Output: tensor of n_hot encoding (atomicNum is index, value is how many of that atom are in the empirical formula)
    ->        standardized to have ratios (as in empirical formula)
    """
    def to_n_hot(self, formulaStr, by_mass=False):
        parsed_formula_dict = formula_to_composition(formulaStr) # maps atomic number to quantity
        formula_tensor = torch.zeros(119)
        for atomic_num, quantity in parsed_formula_dict.items():
            formula_tensor[atomic_num] = quantity
            if by_mass:
                # 0-indexing vs 1-indexing, verified from https://pythonhosted.org/chempy/_modules/chempy/util/parsing.html#mass_from_composition
                formula_tensor[atomic_num] *= relative_atomic_masses[atomic_num - 1] 
        formula_tensor = 1.0 / formula_tensor.sum() * formula_tensor # standardize (just hold ratios)
        return formula_tensor

    """
    -> Input: 
        -> formulaStr of form 'A2(B3C5)3'
        -> standardized (whether or not to make elements sum to 1)
        -> numToRemove (how many elements to randomly exclude)
    -> Output: n-hot tensor encoding (atomicNum is index, value is 1 if that atom is in the empirical formula, 0 otherwise) -
    ->        can be standardized or not
    """
    def to_composition_vector_no_ratio(self, formulaStr, standardized=False, numToRemove=0):
        parsed_formula_dict = formula_to_composition(formulaStr) # maps atomic number to quantity
        formula_tensor = torch.zeros(119)

        elements = [atomic_num for atomic_num in parsed_formula_dict]
        elements = random.sample(elements, max(1, len(elements) - numToRemove)) # sample at least 1 element

        for atomic_num in elements:
            formula_tensor[atomic_num] = 1
        if standardized:
            formula_tensor = 1.0 / formula_tensor.sum() * formula_tensor # standardize
        return formula_tensor        
    
    """
    Input:
    -> name of form "CHGCAR_mp-xxx.npy"
    Return:
    -> "mp-xxx"
    """
    def get_mp_id(self, name):
        without_ext = name.split('.')[0]
        without_prefix = without_ext.split('_')[1]
        assert 'mp-' in without_prefix
        assert '.' not in without_prefix
        assert 'CHGCAR_' not in without_prefix
        return without_prefix

    """
    Loads the charge density and xrd vector for crystal [name] in directory [root]
    Also stores the chemical formula, unit cell parameters, and spacegroup
    Also updates self.id_to_charge_densities
    """
    def process_crystal(self, root_plus_name):
        root, name = root_plus_name

        # get corresponding xrd vector
        success = self.store_corresponding_xrd_vector(name)
        # make sure this data exists before we keep going
        if not success:
            print('xrd does not exist for {}'.format(name))
            return

        # store chemical formula
        mp_id = self.get_mp_id(name)
        formula = self.id_to_formula[mp_id]
        self.formulas.append(formula)

        # store lattice
        lattice = self.id_to_flattened_lattices[mp_id]
        self.lattices.append(lattice)

        # store spacegroup
        spacegroup_one_hot = self.id_to_spacegroup_one_hot[mp_id]
        self.spacegroups_one_hot.append(spacegroup_one_hot)
        self.spacegroups_int.append(self.id_to_spacegroup[mp_id])

        # store filepath
        curr_filepath = os.path.join(root, name)
        self.filepaths.append(curr_filepath)
        # store corresponding charge density map
        curr_charge_density = np.load(curr_filepath)
        self.charge_densities.append(curr_charge_density)
        # store dimension of the charge density map
        curr_map_size = curr_charge_density.shape
        self.map_sizes.append(curr_map_size)
        # update id->charge density
        self.id_to_charge_density[mp_id] = curr_charge_density

        return

    """
    -> charge_filename = the .npy filename of the charge density, exe: CHGCAR_mp-xxxx.npy
    -> root = the directory in which the xrd file was found, 
        exe: ~/data/gabeguo/crystallography/xrd_data__moka_cuka/train
    Return:
    -> true if successful, false if failed
    Post:
    -> self.xrd_vectors contains the corresponding multi-channel XRD diffraction pattern to this charge density
    """
    def store_corresponding_xrd_vector(self, charge_filename):
        xrd_filename = 'diffraction_peaks_{}.pt'.format(self.get_mp_id(charge_filename))
        filepath = os.path.join(self.xrd_data_dir, xrd_filename)
        
        # this does not exist - can't do
        if not os.path.exists(filepath):
            print('{} does not exist'.format(filepath))
            return False

        peak_data = torch.load(filepath)
        assert peak_data.shape[1] == 1024
        assert peak_data.shape[1] > peak_data.shape[0]
        if self.num_channels < peak_data.shape[0]:
            #print('masking')
            peak_data[self.num_channels:] = 0 # mask out the unused channels
        self.xrd_vectors.append(peak_data.to(dtype=torch.float32))

        return True

    def __len__(self):
        return len(self.filepaths)

    """
    Input:
    -> index = crystal number (arbitrary)
    Returns:
    -> (self.num_channels, 1024)-dim vector of charge densities for a given crystal
    -> 119-dimensional vector of empirical chemical formula encoding for the given crystal
    -> 9-dimensional flattened lattice vectors for the given crystal
    -> 231-dimensional vector of spacegroup for the given crystal
    -> (NUM_X_SAMPLES * NUM_Y_SAMPLES * NUM_Z_SAMPLES) coordinates: multiple (x, y, z) from the same crystal by bin sampling
    -> (NUM_X_SAMPLES * NUM_Y_SAMPLES * NUM_Z_SAMPLES) charge densities corresponding to the (x, y, z)'s for this crystal
    """
    def __getitem__(self, idx):
        # get the charge density map and its dimensions
        curr_density_map = self.charge_densities[idx]
        curr_dims = self.map_sizes[idx]
        # get the (self.num_channels, 1024)-dim charge vector
        curr_xrd_vector = self.xrd_vectors[idx]
        assert curr_xrd_vector.shape[1] == 1024
        assert curr_xrd_vector.shape[1] > curr_xrd_vector.shape[0]
        # get the empirical formula vector
        curr_chem_formula = self.formulas[idx]
        assert curr_chem_formula.shape[0] == 119
        # get the concatenated lattice vectors
        curr_unit_cell = self.lattices[idx]
        assert curr_unit_cell.shape == (9,)
        # get the spacegroup one-hot
        curr_spacegroup = self.spacegroups_one_hot[idx]
        assert curr_spacegroup.shape == (231,)

        coordinates = list()
        charges = list()

        # sample charge density
        for x in range(self.NUM_X_SAMPLES):
            for y in range(self.NUM_Y_SAMPLES):
                for z in range(self.NUM_Z_SAMPLES):
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
                    # make the proportional x, y, z exact
                    proportional_x = scaled_x / curr_dims[0]
                    proportional_y = scaled_y / curr_dims[1]
                    proportional_z = scaled_z / curr_dims[2]

                    # get the current charge
                    curr_charge = curr_density_map[scaled_x, scaled_y, scaled_z]

                    # add to running list
                    coordinates.append([proportional_x, proportional_y, proportional_z])
                    charges.append(curr_charge)

        coordinates = torch.tensor(coordinates).to(dtype=torch.float32)
        assert coordinates.shape == (self.NUM_X_SAMPLES * self.NUM_Y_SAMPLES * self.NUM_Z_SAMPLES, 3)
        charges = torch.tensor(charges).to(dtype=torch.float32)
        assert charges.shape == (self.NUM_X_SAMPLES * self.NUM_Y_SAMPLES * self.NUM_Z_SAMPLES,)

        return curr_xrd_vector, curr_chem_formula, curr_unit_cell, curr_spacegroup, \
                coordinates, charges

    """
    Returns all the charge density maps in spacegroup
    """
    def get_crystals_in_spacegroup(self, spacegroup):
        mpids_in_spacegroup = self.spacegroup_to_mpids[spacegroup]
        crystals_in_spacegroup = [self.id_to_charge_density[mpid] for mpid in mpids_in_spacegroup \
                                  if mpid in self.id_to_charge_density]
        return crystals_in_spacegroup