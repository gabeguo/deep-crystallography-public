from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from emmet.core.summary import HasProps
from emmet.core.symmetry import CrystalSystem

from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import io
import os
import sys
sys.path.append('../')
import json
import argparse
import numpy as np

from pyrho.charge_density import ChargeDensity

from multiprocessing import pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import torch

import matplotlib.pyplot as plt

#from constants import *
API_KEY = 'fill in' # TODO
XRD_VECTOR_DIM = 1024
POSSIBLE_CRYSTAL_SYSTEMS = {"Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"}
UNSTABLE = -1
STABLE = 1
ANY_STABILITY = 0

def write_npy(charge_density, output_filepath, args):
    # reshape
    charge_density = ChargeDensity.from_pmg(charge_density)
    #print(charge_density.grid_shape)
    upsample_factor = args.sample_freq // min(charge_density.grid_shape) + 1
    assert upsample_factor >= 1
    charge_density = charge_density.get_transformed(sc_mat=np.eye(3), \
                                                    grid_out=[args.sample_freq, args.sample_freq, args.sample_freq], \
                                                    up_sample=upsample_factor)
    # normalize
    normalized_data = charge_density.normalized_data['total'] # e-/Ang^3
    # save as numpy array
    np.save(output_filepath, normalized_data)
    return

def save_crystal(imat):
    the_filepath = os.path.join(args.output_dst_charge, f"CHGCAR_{imat.material_id}.npy")
    
    # skip if already found
    if os.path.exists(the_filepath):
        # make sure we have xrd
        save_xrd(imat)
        return
    
    # stop if we aren't downloading new crystals
    if args.no_new_crystals:
        print('Don\'t need it: {}'.format(imat.material_id))
        return
    
    print(f"Searching for charge: {imat.material_id}")
    try:
        with MPRester(API_KEY) as mpr:
            chgcar = mpr.get_charge_density_from_material_id(imat.material_id)
        if chgcar is None:
            print('\tfailure null chgcar: {}'.format(imat.material_id))
            return
        write_npy(charge_density=chgcar, output_filepath=the_filepath, args=args)
        print('\tsuccess charge: {}'.format(imat.material_id))
    except Exception as the_error:
        print('\tfailure charge MPRestError: {} {}'.format(imat.material_id, the_error))
        return
    
    save_xrd(imat)

    return

def save_xrd(imat):
    the_filepath = os.path.join(args.output_dst_xrd, f"diffraction_peaks_{imat.material_id}.pt")
    
    # skip if already found
    if os.path.exists(the_filepath):
        return
            
    #print(f"Searching for xrd: {imat.material_id}")
    try:
        with MPRester(API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(imat.material_id)

        sga = SpacegroupAnalyzer(structure)
        conventional_structure = sga.get_conventional_standard_structure()

        # these patterns are calculated on-the-fly from the structure
        all_patterns = list()
        for curr_wavesource in wavesources_in_use:
            curr_wavelength = WAVELENGTHS[curr_wavesource]
            calculator = XRDCalculator(wavelength=curr_wavelength)
            pattern = calculator.get_pattern(conventional_structure, two_theta_range=(args.min_theta, args.max_theta))
            all_patterns.append(pattern)
        assert len(all_patterns) == len(wavesources_in_use)

        patterns_as_tensor = create_xrd_tensor(all_patterns)
        torch.save(patterns_as_tensor, the_filepath)

        print('\tsuccess XRD: {}'.format(imat.material_id))
    except MPRestError as the_err:
        print('\tfailure XRD MPRestError: {} {}'.format(imat.material_id, the_err))
        return
    except ValueError:
        print('\tfailure XRD ValueError: {}'.format(imat.material_id))
        return
    return

"""
Input: 
-> the_patterns: list; XRD patterns, where the_patterns[i] = the XRD data calculated for wavesources_in_use[i], directly from XRDCalculator
Output
-> pattern_tensor: tensor of shape (len(wavesources_in_use), XRD_VECTOR_DIM); where pattern_tensor[i, :] = an XRD strip pattern for wavesources_in_use[i];
                    works with PyTorch 1d convolutions that have (N, channels, sequence length)
"""
def create_xrd_tensor(the_patterns):
    assert XRD_VECTOR_DIM == 1024
    peak_data = torch.zeros((len(the_patterns), XRD_VECTOR_DIM)) # C x L
    for idx in range(len(the_patterns)):
        pattern = the_patterns[idx]
        peak_locations = pattern.x.tolist()
        peak_values = pattern.y.tolist()

        assert len(peak_locations) == len(peak_values)

        assert args.min_theta == 0
        assert args.max_theta == 180

        for i2 in range(len(peak_locations)):
            theta = peak_locations[i2]
            height = peak_values[i2] / 100
            scaled_location = int(XRD_VECTOR_DIM * theta / (args.max_theta - args.min_theta))
            peak_data[idx, scaled_location] = max(peak_data[idx, scaled_location], height) # just in case really close

        # # show: debug
        # plt.scatter([i for i in range(len(peak_data[idx]))], peak_data[idx], s=2)
        # plt.show()

    return peak_data

def process_files():
    print(API_KEY)
    # Thanks https://github.com/materialsproject/pyrho/blob/main/notebooks/download_from_api.ipynb

    with MPRester(API_KEY) as mpr:
        assert args.crystal_system is None or args.crystal_system in POSSIBLE_CRYSTAL_SYSTEMS
        stable_materials = mpr.summary.search(\
            has_props=[HasProps.charge_density], 
            is_stable=(True if args.stability == STABLE else (False if args.stability == UNSTABLE else None)),
            crystal_system=args.crystal_system,
            fields=["material_id", "formula_pretty", 'symmetry', 'structure']\
        )
    # print(stable_materials[0].symmetry.number)
    # print(stable_materials[0].structure.lattice)
    print(f"The query returned {len(stable_materials)} documents.")
    # write json
    if args.record_id_to_formula: # mpid to chemical formula (empirical)
        with open(os.path.join(args.output_dst_charge, '{}_formulas.json'.format(args.crystal_system)), 'w') as fout:
            fout.write(json.dumps({imat.material_id:imat.formula_pretty for imat in stable_materials}, indent=6))

    if args.record_id_to_spacegroup: # mpid to international space group number
        with open(os.path.join(args.output_dst_charge, '{}_space_groups.json'.format(args.crystal_system)), 'w') as fout:
            fout.write(json.dumps({imat.material_id:imat.symmetry.number for imat in stable_materials}, indent=6))

    if args.record_id_to_lattice: # mpid to crystal lattice vectors (x0, y0, z0, x1, y1, z1, x2, y2, z2)
        with open(os.path.join(args.output_dst_charge, '{}_lattice_vectors.json'.format(args.crystal_system)), 'w') as fout:
            fout.write(json.dumps({imat.material_id:imat.structure.lattice._matrix.tolist() for imat in stable_materials}, indent=6))
    
    # terminate, if we were just trying to get that easy info
    if args.record_id_to_formula or args.record_id_to_spacegroup or args.record_id_to_lattice:
        return

    for item in tqdm(stable_materials):
        save_crystal(item)
    pool = ThreadPool(processes=8)
    for _ in tqdm(pool.imap(save_crystal, stable_materials), total=len(stable_materials)):
        pass
    
    return

def main():
    parser = argparse.ArgumentParser(description='Download charge densities as .npy')
    parser.add_argument('--output_dst_charge', type=str, default='/data/therealgabeguo/crystallography/default_charge_dir', metavar='CDD',
                        help='where is the charge density data going to be saved?')
    parser.add_argument('--output_dst_xrd', type=str, default='/data/therealgabeguo/crystallography/default_xrd_dir', metavar='XRD',
                        help='where is the xrd data going to be saved?')
    parser.add_argument('--sample_freq', type=int, default=50,
                        help='how many charge density samples do you want per axis (x, y, z; same for all)')
    parser.add_argument('--stability', type=int, default=ANY_STABILITY,
                        help='{} for unstable, {} for stable, {} for any stability'.format(UNSTABLE, STABLE, ANY_STABILITY))
    parser.add_argument('--min_theta', type=int, default=0,
                        help='What is the min theta for XRD?')
    parser.add_argument('--max_theta', type=int, default=180,
                        help='What is the max theta for XRD?')
    parser.add_argument('--wave_sources', type=str, default='CuKa MoKa CrKa FeKa CoKa AgKa',
                        help='What are the desired wave sources?')
    # TODO: refactor
    parser.add_argument('--no_new_crystals', action='store_true', default=False,
                        help='Do not download new crystals: only get the XRD patterns for those charge densities that are missing corresponding XRD')
    parser.add_argument('--record_id_to_formula', action='store_true', default=False,
                        help='Does not actually download any charge density or XRD information. ' + \
                            'Writes .json file mapping crystal id to chemical formula for the materials in the specified crystal system.')
    parser.add_argument('--record_id_to_spacegroup', action='store_true', default=False,
                        help='Does not actually download any charge density or XRD information. ' + \
                            'Writes .json file mapping crystal id to international spacegroup number for the materials in the specified crystal system.') 
    parser.add_argument('--record_id_to_lattice', action='store_true', default=False,
                        help='Does not actually download any charge density or XRD information. ' + \
                            'Writes .json file mapping crystal id to nine-dimensional concatenated lattice vector for the materials in the specified crystal system.')
    parser.add_argument('--crystal_system', type=str, default=None,
                        help='What crystal system to download data for. Options: None (default) or {}'.format(POSSIBLE_CRYSTAL_SYSTEMS))
    
    global args
    global wavesources_in_use

    args = parser.parse_args()
    wavesources_in_use = args.wave_sources.split()
    for item in wavesources_in_use:
        assert item in WAVELENGTHS

    os.makedirs(args.output_dst_charge, exist_ok=True)
    os.makedirs(args.output_dst_xrd, exist_ok=True)

    with open(os.path.join(args.output_dst_xrd, 'metadata.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    process_files()

    return

if __name__ == "__main__":
    main()