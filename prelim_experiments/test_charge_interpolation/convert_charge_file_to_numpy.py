import os
import argparse
import numpy as np

from pyrho.charge_density import ChargeDensity

import sys
sys.path.append('../../')
from constants import *

"""
Literally replaces the extension with .npy
"""
def convert_filepath_to_npy(filepath):
    return os.path.splitext(filepath)[0] + '.npy'

"""
Input:
-> args: contains destination directory (args.output_dst)
-> charge_filepath: where the .vasp file is stored
-> rel_filepath: the relative filepath of charge_filepath to args.charge_data_src
Outcome:
-> Saves charge_filepath as a .npy file to args.output_dst under rel_filepath (changed to .npy, not .vasp)
"""
def save_as_numpy_single(args, charge_filepath, rel_filepath):
    npy_rel_filepath = convert_filepath_to_npy(rel_filepath)
    output_filepath = os.path.join(args.output_dst, npy_rel_filepath)
    if os.path.exists(output_filepath):
        # no need to do again
        return

    charge_density = ChargeDensity.from_file(charge_filepath)
    # reshape
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

"""
Saves the charge data as numpy files with parallel directory structure
"""
def save_as_numpy(args):
    # create output dir
    for category in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dst, category), exist_ok=True)
    # loop through charge data
    for root, _, files in os.walk(args.charge_data_src):
        for filename in files:
            if '.vasp' not in filename or 'mp' not in filename:
                continue
            charge_filepath = os.path.join(root, filename)
            rel_filepath = os.path.relpath(charge_filepath, args.charge_data_src)
            # save the data as numpy
            save_as_numpy_single(args=args, charge_filepath=charge_filepath, rel_filepath=rel_filepath)

    return

def main():
    parser = argparse.ArgumentParser(description='Save the charge files as .npy for more compressed storage')
    parser.add_argument('--charge_data_src', type=str, default=CHARGE_DATA_DIR, metavar='CDS',
                        help='where is the charge data stored')
    parser.add_argument('--output_dst', type=str, default='/data/therealgabeguo/crystallography/charge_data_npy', metavar='CDD',
                        help='where is the new .npy file going to be saved?')
    parser.add_argument('--sample_freq', type=int, default=50,
                        help='how many charge density samples do you want per axis (x, y, z; same for all)')
    args = parser.parse_args()

    # remove trailing slash
    if args.charge_data_src[-1] == '/':
        args.charge_data_src = args.charge_data_src[:-1]

    save_as_numpy(args)

    return

if __name__ == "__main__":
    main()