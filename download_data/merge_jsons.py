# Thanks ChatGPT-4 (got starter code from there)

import os
import json
import glob

def merge_json_files(directory, file_wild_card="*_formulas.json", output_name="allMergedFormulas.json", 
                     desired_keywords=None, undesired_keywords=None):
    data = {}

    json_files = glob.glob(os.path.join(directory, file_wild_card))
    if desired_keywords is not None:
        json_files = [file for file in json_files 
                      if any(keyword in file for keyword in desired_keywords)]
    if undesired_keywords is not None:
        json_files = [file for file in json_files 
                      if all(keyword not in file for keyword in undesired_keywords)]
    print('merging: {}'.format(json_files))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            file_data = json.load(f)
        # Merge file_data into data
        for key, value in file_data.items():
            if key not in data:
                data[key] = value
            else:
                print('{} already seen'.format(key))

    # Write merged data to output file
    with open(os.path.join(directory, output_name), 'w') as f:
        json.dump(data, f, indent=4)

# Use the function
desired_crystal_systems = ['Trigonal', 'Tetragonal']
undesired_crystal_systems = ['Cubic', 'Orthorhombic', 'Hexagonal', 'Monoclinic', 'Triclinic']
merge_json_files('/data/therealgabeguo/crystallography/crystal_systems_all', 
                 file_wild_card="*_formulas.json", output_name="TrigonalTetragonal_formulas.json", 
                 desired_keywords=desired_crystal_systems, undesired_keywords=undesired_crystal_systems)
merge_json_files('/data/therealgabeguo/crystallography/crystal_systems_all', 
                 file_wild_card="*_lattice_vectors.json", output_name="TrigonalTetragonal_lattice_vectors.json",
                 desired_keywords=desired_crystal_systems, undesired_keywords=undesired_crystal_systems)
merge_json_files('/data/therealgabeguo/crystallography/crystal_systems_all', 
                 file_wild_card="*_space_groups.json", output_name="TrigonalTetragonal_space_groups.json",
                 desired_keywords=desired_crystal_systems, undesired_keywords=undesired_crystal_systems)
# merge_json_files('/data/therealgabeguo/crystallography/crystal_systems_all', 
#                  file_wild_card="*_formulas.json", output_name="allMergedFormulas.json")
# merge_json_files('/data/therealgabeguo/crystallography/crystal_systems_all', 
#                  file_wild_card="*_lattice_vectors.json", output_name="allMergedLatticeVectors.json")
# merge_json_files('/data/therealgabeguo/crystallography/crystal_systems_all', 
#                  file_wild_card="*_space_groups.json", output_name="allMergedSpaceGroups.json")