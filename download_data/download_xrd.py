from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from emmet.core.summary import HasProps
import os
import sys
sys.path.append('../')
from constants import *
import json

from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def find_split(material_id):
    base_path = '/home/gabeguo/data/crystallography/charge_data_npy'
    for modality in ['train', 'val', 'test']:
        modality_path = os.path.join(base_path, modality, 'CHGCAR_{}.npy'.format(material_id))
        if os.path.exists(modality_path):
            return modality
    print('{} not found in charge data'.format(material_id))
    return None

def main(wave_source, min_theta, max_theta):
    print(API_KEY)
    
    with MPRester(API_KEY) as mpr:
        stable_materials = mpr.summary.search(\
            #has_props=[HasProps.charge_density], 
            is_stable=True,
            fields=["material_id"]\
        )
        print(f"The query returned {len(stable_materials)} documents.")

    curr_wavelength = WAVELENGTHS[wave_source]

    DATA_DIR = '/home/gabeguo/data/crystallography/xrd_data_json/{}'.format(wave_source)
    for the_split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATA_DIR, the_split), exist_ok=True)

    found_xrd_ids = []
    failed_xrd_ids = []

    with MPRester(API_KEY) as mpr:
        for imat in stable_materials:
            the_split = find_split(imat.material_id) # TODO: implement
            if the_split is None:
                continue
            the_filepath = os.path.join(DATA_DIR, the_split, f"diffraction_peaks_{imat.material_id}.json")
            
            # skip if already found
            if os.path.exists(the_filepath):
                continue
            
            print(f"Searching for {imat.material_id}")
            try:
                structure = mpr.get_structure_by_material_id(imat.material_id)

                sga = SpacegroupAnalyzer(structure)
                conventional_structure = sga.get_conventional_standard_structure()

                # this example shows how to obtain an XRD diffraction pattern
                # these patterns are calculated on-the-fly from the structure
                calculator = XRDCalculator(wavelength=curr_wavelength)
                pattern = calculator.get_pattern(conventional_structure, two_theta_range=(min_theta, max_theta))

                # This gets the peak data
                #print('\tpeak locations:', pattern.x)
                #print('\tpeak values:', pattern.y)

                json_output = dict()
                json_output['wavelength'] = curr_wavelength
                json_output['min_theta'] = min_theta
                json_output['max_theta'] = max_theta
                json_output['peak_locations'] = pattern.x.tolist()
                json_output['peak_values'] = pattern.y.tolist()

                with open(the_filepath, 'w') as fout:
                    fout.write(json.dumps(json_output))
                #     fout.write('wavelength: {}\n'.format(curr_wavelength))
                #     fout.write('min theta: {}\n'.format(min_theta))
                #     fout.write('max theta: {}\n'.format(max_theta))
                #     fout.write('peak locations: {}\n'.format(pattern.x))
                #     fout.write('peak values: {}\n'.format(pattern.y))

                found_xrd_ids.append(imat.material_id)
                print('\tsuccess: {}, {}'.format(imat.material_id, len(found_xrd_ids)))
            except MPRestError:
                failed_xrd_ids.append(imat.material_id)
                print('\tfailure: {}'.format(imat.material_id))
            except ValueError:
                failed_xrd_ids.append(imat.material_id)
                print('\tfailure: {}'.format(imat.material_id))
    return

if __name__ == "__main__":
    wave_source = 'CuKa'
    min_theta = 0
    max_theta = 180

    main(wave_source=wave_source, min_theta=min_theta, max_theta=max_theta)
