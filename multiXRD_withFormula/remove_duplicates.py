import json
import argparse
from tqdm import tqdm
import os

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Remove Duplicates')
    parser.add_argument('--old_mp_id_to_formula', type=str, default='/home/gabeguo/data/crystallography_paper_version/crystal_systems_all/Trigonal_formulas.json',
                        help='Input JSON file with mapping of mp_id to chemical formula.')
    parser.add_argument('--old_mp_id_to_spacegroup', type=str, default='/home/gabeguo/data/crystallography_paper_version/crystal_systems_all/Trigonal_space_groups.json',
                        help='Input JSON file with mapping of mp_id to spacegroup numbers.')
    parser.add_argument('--new_mp_id_to_formula', type=str, default='/home/gabeguo/data/crystallography_paper_version/crystal_systems_all/Trigonal_formulas_NO_DUPLICATES.json',
                        help='Output JSON file with mapping of mp_id to chemical formula, with duplicates removed')
    parser.add_argument('--new_mp_id_to_spacegroup', type=str, default='/home/gabeguo/data/crystallography_paper_version/crystal_systems_all/Trigonal_space_groups_NO_DUPLICATES.json',
                        help='Output JSON file with mapping of mp_id to spacegroup numbers, with duplicates removed')
    
    args = parser.parse_args()

    # mapping of MP-ID to formula & spacegroup
    with open(args.old_mp_id_to_formula, 'r') as formula_file:
        mp_id_to_formula = json.load(formula_file)
    with open(args.old_mp_id_to_spacegroup, 'r') as spacegroup_file:
        mp_id_to_spacegroup = json.load(spacegroup_file)

    # cross-check for duplicates
    seen_combos = dict()
    for mp_id in tqdm(mp_id_to_formula):
        curr_combo = (mp_id_to_formula[mp_id], mp_id_to_spacegroup[mp_id])
        if curr_combo not in seen_combos:
            seen_combos[curr_combo] = list()
        seen_combos[curr_combo].append(mp_id)

    # create new outputs
    assert args.old_mp_id_to_formula != args.new_mp_id_to_formula
    assert args.old_mp_id_to_spacegroup != args.new_mp_id_to_spacegroup
    new_mpid_to_formula = dict()
    new_mpid_to_spacegroup = dict()
    for curr_combo in seen_combos:
        curr_formula, curr_spacegroup = curr_combo
        first_mpid = seen_combos[curr_combo][0] # just choose one - remove all the other duplicates
        new_mpid_to_formula[first_mpid] = curr_formula
        new_mpid_to_spacegroup[first_mpid] = curr_spacegroup
    
    # write new outputs
    with open(args.new_mp_id_to_formula, 'w') as fout:
        fout.write(json.dumps(new_mpid_to_formula, indent=4))
    with open(args.new_mp_id_to_spacegroup, 'w') as fout:
        fout.write(json.dumps(new_mpid_to_spacegroup, indent=4))
    
    print('total num items:', len(seen_combos))
    print('done')
    return

if __name__ == "__main__":
    main()