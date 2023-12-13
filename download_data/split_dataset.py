import os
import argparse
import shutil
import random
import json

import sys
sys.path.append('../../')

SPLIT = '_split'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
CHARGE = 'charge'
XRD = 'xrd'

"""
Returns mp-xxx
given filename of format CHGCAR_mp-xxx.npy
or diffraction_peaks_mp-xxx.pt
"""
def get_mp_id(charge_filename):
    assert '.pt' in charge_filename or '.npy' in charge_filename
    filename_base = charge_filename.split('.')[0]
    mp_id = filename_base.split('_')[-1]
    assert 'mp-' in mp_id
    assert '.pt' not in mp_id
    assert '.npy' not in mp_id
    assert len(mp_id) > len('mp-')
    return mp_id

"""
Input:
-mp_id in format mp-xxx
-xrd_filepaths contains all the filepaths for xrd data,
indexed by mp_id:filepath
Returns:
-xrd filepath corresponding to this mp_id
-None if that can't be found
"""
def find_xrd_filepath_from_mp(mp_id, xrd_filepaths):
    if mp_id in xrd_filepaths:
        return xrd_filepaths[mp_id]
    return None

"""
Returns:
-TRAIN, VAL, or TEST depending on args.train_percent, args.val_percent, args.test_percent
-By default, all unstable elements go into TRAIN
"""
def get_data_split_category(args, stable):
    the_num = random.random() * 100
    if the_num < args.train_percent or (not stable):
        return TRAIN
    if the_num < args.train_percent + args.val_percent:
        return VAL
    return TEST

"""
Checks if mp_id is a key in the dict of stable_elems (which is mp_id to formula)
"""
def is_stable(mp_id, stable_elems):
    return mp_id in stable_elems

"""
Moves the data in charge_path and xrd_path to a new filepath.
Assigns it to train, test, val with probability,
dictated by args.train_percent, args.val_percent, args.test_percent.
Note that all unstable elements are automatically in train.
Only does it for one at a time (not list).
"""
def move_data_to_new_dest(args, charge_path, xrd_path, stable_elems):
    data_split_category = get_data_split_category(args, stable=is_stable(get_mp_id(charge_path), stable_elems))
    for (orig_path, new_dir) in zip([charge_path, xrd_path], [args.charge_data_dst, args.xrd_data_dst]):
        fname = orig_path.split('/')[-1]
        assert '.' in fname
        new_path = os.path.join(new_dir, data_split_category, fname)
        if os.path.exists(new_path):
            continue # skip if already been done
        #print(orig_path, '\n\t', new_path)
        #shutil.copyfile(src=orig_path, dst=new_path)
        os.rename(src=orig_path, dst=new_path)
    return

"""
Splits the crystallography dataset.
Splits both charge and xrd folders.
Makes sure same items are in same categories in both.
Copies the items into new folders.
Makes list of excluded items.
"""
def split_data(args):
    # list of exclusions
    exclusions = list()
    # make output directories
    for the_dir in [args.charge_data_dst, args.xrd_data_dst]:
        os.makedirs(the_dir, exist_ok=True)
        # split into train val test
        for category in [TRAIN, VAL, TEST]:
            os.makedirs(os.path.join(the_dir, category), exist_ok=True)
    # create xrd data
    xrd_filepaths = {get_mp_id(filename):os.path.join(root, filename) \
                    for (root, dirs, files) in os.walk(args.xrd_data_src) \
                    for filename in files \
                    if '.pt' in filename and '_mp-' in filename}
    # log stable elems (key: mpid, val: formula)
    with open(args.stable_elems, 'r') as fin:
        stable_elems = json.load(fin)
    # loop through charge data
    count_used = 0
    total_count = 0
    for root, _, files in os.walk(args.charge_data_src):
        for filename in files:
            if '.npy' not in filename or 'mp' not in filename:
                continue
            total_count += 1
            mp_id = get_mp_id(filename)
            charge_filepath = os.path.join(root, filename)
            # find corresponding xrd data
            xrd_filepath = find_xrd_filepath_from_mp(mp_id, xrd_filepaths)
            if not os.path.exists(xrd_filepath) or xrd_filepath is None:
                exclusions.append(charge_filepath)
                continue
            # move the data to new destination
            move_data_to_new_dest(args=args, charge_path=charge_filepath, 
                                  xrd_path=xrd_filepath, stable_elems=stable_elems)
            count_used += 1
    # print exclusions & usage
    exclusion_filepath = os.path.join(args.charge_data_dst, 'excluded_IDs.txt')
    with open(exclusion_filepath, 'w') as fout:
        for line in exclusions:
            fout.write(str(line) + '\n')
    # print number of items used
    print('{} out of {} possible charge densities have been used'.format(count_used, total_count))
    
    return

def main():
    parser = argparse.ArgumentParser(description='Split the Crystallography Dataset')
    parser.add_argument('--charge_data_src', type=str, default='/home/gabeguo/data/replicate_crystallography/trigonal/charges', metavar='CDS',
                        help='where is the charge data stored')
    parser.add_argument('--xrd_data_src', type=str, default='/home/gabeguo/data/replicate_crystallography/trigonal/xrds', metavar='XDS',
                        help='where is the XRD data stored')
    parser.add_argument('--charge_data_dst', type=str, default='/home/gabeguo/data/replicate_crystallography/trigonal_split/charges', metavar='CDD',
                        help='where is the charge data going to be moved')
    parser.add_argument('--xrd_data_dst', type=str, default='/home/gabeguo/data/replicate_crystallography/trigonal_split/xrds', metavar='XDD',
                        help='where is the XRD data going to be moved')
    parser.add_argument('--stable_elems', type=str, default='/home/gabeguo/data/replicate_crystallography/STABLE_material_info/Trigonal_formulas.json',
                        help='filepath to json containing all the stable elements as keys (values are irrelevant)')
    parser.add_argument('--train_percent', type=int, default=80, metavar='TrP',
                        help='percentage of stable data used for training (all unstable forced to train)')
    parser.add_argument('--val_percent', type=int, default=10, metavar='VP',
                        help='percentage of stable data used for validation (all unstable forced to train)')
    args = parser.parse_args()

    # remove trailing slash
    if args.charge_data_src[-1] == '/':
        args.charge_data_src = args.charge_data_src[:-1]
    if args.xrd_data_src[-1] == '/':
        args.xrd_data_src = args.xrd_data_src[:-1]

    # make sure we have valid input
    assert args.train_percent + args.val_percent <= 100 \
            and args.train_percent + args.val_percent >= 0 \
            and args.train_percent >= 0 \
            and args.val_percent >= 0
    assert os.path.exists(args.charge_data_src)
    assert os.path.exists(args.xrd_data_src)

    test_percent = 100 - args.train_percent - args.val_percent

    parser.add_argument('--test_percent', type=int, default=test_percent, metavar='TP',
                        help='percentage of data used for testing')
    args = parser.parse_args()

    split_data(args)

    return

# Works!
if __name__ == "__main__":
    main()
