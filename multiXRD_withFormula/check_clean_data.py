import json
import argparse
from tqdm import tqdm
import os
from chempy.util.parsing import formula_to_composition

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train the Charge Density Estimation Net')
    parser.add_argument('--mp_id_to_formula', type=str, default='/home/gabeguo/data/crystallography_paper_version/crystal_systems_all/Trigonal_formulas_NO_DUPLICATES.json',
                        help='JSON file with mapping of mp_id to chemical formula.')
    parser.add_argument('--mp_id_to_spacegroup', type=str, default='/home/gabeguo/data/crystallography_paper_version/crystal_systems_all/Trigonal_space_groups_NO_DUPLICATES.json',
                        help='JSON file with mapping of mp_id to spacegroup numbers.')
    parser.add_argument('--test_charge_data_dir', type=str, default='/home/gabeguo/data/crystallography_paper_version/charge_data_npy/test',
                        help='where is the charge data stored')
    parser.add_argument('--test_xrd_data_dir', type=str, default='/home/gabeguo/data/crystallography_paper_version/xrd_data_tensor__moka_crka/test',
                        help='where is the xrd data stored')
    
    args = parser.parse_args()

    testing_mpids = set()
    # Get all the testing crystals
    for root, dirs, files in os.walk(args.test_charge_data_dir):
        for file in files:
            # format CHGCAR_mp-xx.npy
            mpid = file.split('_')[1][:-4]
            testing_mpids.add(mpid)

    # cross-check for duplicates
    seen_combos = dict()

    # mapping of MP-ID to formula & spacegroup
    with open(args.mp_id_to_formula, 'r') as formula_file:
        mp_id_to_formula = json.load(formula_file)
    with open(args.mp_id_to_spacegroup, 'r') as spacegroup_file:
        mp_id_to_spacegroup = json.load(spacegroup_file)

    # make all the testing combos
    testing_combos = dict()
    for mp_id in tqdm(mp_id_to_formula):
        if mp_id not in testing_mpids:
            continue
        # impose canonical order on formula
        ordered_formula = tuple(sorted(formula_to_composition(mp_id_to_formula[mp_id]).items()))
        curr_combo = (ordered_formula, mp_id_to_spacegroup[mp_id])
        testing_combos[curr_combo] = mp_id

    # check which testing combos are duplicated
    duplicated_testing_combos = set()
    for mp_id in tqdm(mp_id_to_formula):
        curr_combo = (tuple(sorted(formula_to_composition(mp_id_to_formula[mp_id]).items())), 
                        mp_id_to_spacegroup[mp_id])
        if curr_combo in testing_combos and mp_id != testing_combos[curr_combo]:
            print(f'{mp_id} duplicates testing item {testing_combos[curr_combo]}: {curr_combo}')
            duplicated_testing_combos.add(curr_combo)
    
    # sanity check
    print('sanity check:')
    for curr_testing_combo, mp_id in list(testing_combos.items())[:10]:
        print(mp_id)
        print('\t', curr_testing_combo)
        print('\t', mp_id_to_formula[mp_id])

    print('num leaked testing items:', len(duplicated_testing_combos))
    print('total num testing items:', len(testing_combos))
    print('done')
    return

if __name__ == "__main__":
    main()