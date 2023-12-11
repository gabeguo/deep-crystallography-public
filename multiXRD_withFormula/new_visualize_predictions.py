import numpy as np

import plotly.graph_objects as go

import os
import sys
sys.path.append('./models')
import torch
import json

import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib
import pandas as pd

from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from chempy.util.parsing import formula_to_composition

from crystal_dataset import CrystalDataset
from crystal_mlp import ChargeDensityRegressor

import argparse
from tqdm import tqdm
from datetime import datetime

"""
Saves a visualization of the charge density map
"""
def plot_charge_density(density_map, range=(0, 1), name='charge_map.png', 
                        is_ground_truth=False, popup=False, multiple_camera_angles=False):
    X, Y, Z = np.mgrid[0:density_map.shape[0], 0:density_map.shape[1], 0:density_map.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=density_map.flatten(),
        isomin=range[0],
        isomax=range[1],
        opacity=0.8, # max opacity
        opacityscale=[(x, np.cbrt(x)) for x in np.linspace(start=0, stop=1, num=25)],
        surface_count=25, 
        colorscale='rainbow'
        ))
    name_base = name.split('.')[0]
    view_angles = [np.pi / 6, np.pi / 4, np.pi / 3] if multiple_camera_angles else [np.pi / 4]
    for view_idx, angle in enumerate(view_angles):
        fig.update_layout(
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis = dict(visible=False)
            ),
            scene_camera = dict(
                eye=dict(x=1.75 * np.cos(angle), y=1.75 * np.sin(angle), z=1.25)
            ), 
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_traces(showscale=is_ground_truth)
        curr_name = f'{name_base}_angle{view_idx}.png'
        fig.write_image(curr_name)
        if popup:
            fig.show()

    print('\tvalues {}:'.format('ground truth' if is_ground_truth else 'predicted'), 
          '\n\t\tmean:', np.mean(density_map), '\n\t\tstd:', np.std(density_map))

    return

"""
Input: multi-channel xrd_pattern: (n_channels x 1024)
"""
def graph_diffraction_pattern(xrd_pattern, name, args):
    assert len(xrd_pattern.shape) == 2
    assert xrd_pattern.shape[0] >= args.num_channels
    assert xrd_pattern.shape[1] == 1024

    individual_xrd_patterns = list()
    # see each individual pattern
    for i in range(args.num_channels):
        curr_pattern = xrd_pattern[i].cpu().numpy()
        assert curr_pattern.shape == (1024,)
        # repeat it 128 times, so it's easier to see
        for j in range(128):
            individual_xrd_patterns.append(curr_pattern)
        indices = np.array([_ for _ in range(curr_pattern.shape[0])])
        plt.figure(figsize=(16, 2))
        plt.plot(indices, curr_pattern, color='red')
        plt.xlim([0, 1024])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(name[:-4] + '_plot.png', bbox_inches='tight')
        plt.close()
    
    visible_xrd_pattern = np.stack(individual_xrd_patterns)
    plt.imsave(name, visible_xrd_pattern, cmap='binary')
    return

"""
Input:
-> metric_by_elements: (118+1)-dimensional list 
    where metric_by_elements[i] = list of specified metric values (e.g., ssim) for any molecule
    containing element with atomic number i
-> name: what to name the output graph
Result:
-> Makes a bar graph of metric_by_elements showing mean and std of that metric for each element
-> Makes a bar graph showing number of molecules containing each element
"""
def graph_metric_by_elements(metric_by_elements, filename, metric_name):
    print('graph {} by elements'.format(metric_name))

    avgs = list()
    stds = list()
    counts = list()
    for atomic_num in range(len(metric_by_elements)):
        if len(metric_by_elements[atomic_num]) > 0:
            avgs.append(np.mean(metric_by_elements[atomic_num]))
            stds.append(np.std(metric_by_elements[atomic_num]))
        else:
            avgs.append(0)
            stds.append(0)
        counts.append(len(metric_by_elements[atomic_num]))
    
    all_possible_atomic_nums = [i for i in range(len(metric_by_elements))]
    plt.clf()

    # graph avg and std of metric
    plt.figure(figsize=(10, 5))
    plt.bar(x=all_possible_atomic_nums, height=avgs, yerr=stds)
    plt.xticks(all_possible_atomic_nums, fontsize=3.5, rotation=90)
    plt.xlabel('Atomic number')
    plt.ylabel(metric_name)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'), dpi=200)
    plt.clf()

    # graph counts of molecules with that element
    plt.figure(figsize=(10, 5))
    plt.bar(x=all_possible_atomic_nums, height=counts)
    plt.xticks(all_possible_atomic_nums, fontsize=3.5, rotation=90)
    plt.xlabel('Atomic number')
    plt.ylabel('Count')
    count_filename = os.path.join(*filename.split('/')[:-1], 'count_by_atomicNum.pdf')
    if '/' in filename:
        count_filename = '/' + count_filename
    plt.savefig(count_filename)
    plt.savefig(count_filename.replace('.pdf', '.png'), dpi=200)
    plt.clf()

    return

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

def center_of_mass(grid):
    """The ChatGPT-4 write this one"""
    total_mass = np.sum(grid)
    indices = np.indices(grid.shape).astype(float)
    com = np.sum(indices * grid, axis=(1, 2, 3)) / total_mass
    return com

"""
Input:
-> all_num_atoms = list of number of atoms in each crystal in testing
-> all_metric = list of calculated metric for each crystal in testing
-> filename = output figure filename
-> metric_name = name of the metric
Result:
-> graphs the performance by number of atoms (binned)
"""
def graph_metric_by_numAtoms(all_num_atoms, all_metric, filename, metric_name, n_bins=10):
    print('graph {} by number of atoms'.format(metric_name))

    plt.clf()

    min_num_atoms = min(all_num_atoms)
    max_num_atoms = max(all_num_atoms)

    binned_data = [[] for i in range(n_bins)]

    for i in range(len(all_num_atoms)):
        curr_num_atoms = all_num_atoms[i]
        curr_metric_val = all_metric[i]

        bin_idx = min(int(n_bins * (curr_num_atoms - min_num_atoms) / max_num_atoms), n_bins - 1)
        binned_data[bin_idx].append(curr_metric_val)
    
    binned_mean = [np.mean(data_in_bin) if len(data_in_bin) > 0  else 0 for data_in_bin in binned_data]
    binned_std = [np.std(data_in_bin) if len(data_in_bin) > 0 else 0 for data_in_bin in binned_data]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(binned_data)), binned_mean, yerr=binned_std)
    plt.xlabel('Number of atoms')
    plt.ylabel(metric_name)
    plt.xticks(ticks=[i for i in range(len(binned_data))], labels=np.round(np.linspace(min_num_atoms, max_num_atoms, n_bins, endpoint=False), 1), rotation=90)  # Rotate x-axis labels for readability
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'), dpi=200)

    plt.clf()

    return

"""
Runs the model's predictions on every crystal in test_loader
"""
def evaluate(model, test_loader, device, args):
    model.eval()

    # general results
    all_ssim = list()
    all_psnr = list()
    mpid_to_ssim = dict()
    mpid_to_psnr = dict()
    mpid_to_com = dict()
    # fine-grained results
    ssim_by_elements = [[] for i in range(118+1)] # ssim_by_elements[i] = list of SSIMs for molecules containing element of periodic number i
    psnr_by_elements = [[] for i in range(118+1)]
    ssim_by_spacegroup = [[] for i in range(230+1)]
    psnr_by_spacegroup = [[] for i in range(230+1)]
    all_num_atoms = list()

    # mpid to formula
    all_mpid_to_formula = dict()

    with torch.no_grad():
        for idx, the_tuple in enumerate(tqdm(test_loader)):
            given_xrd_pattern, chemical_formula, lattice_vector, spacegroup_vector, pos, charge_at_pos = the_tuple
            # put inputs to device
            given_xrd_pattern = given_xrd_pattern.to(device)
            chemical_formula = chemical_formula.to(device)
            lattice_vector = lattice_vector.to(device)
            spacegroup_vector = spacegroup_vector.to(device)
            pos = pos.to(device)
            gt_charges = charge_at_pos.to(device)

            # eliminate data if needed
            if args.num_channels == 0 or args.num_conv_blocks == 0:
                given_xrd_pattern = torch.zeros_like(given_xrd_pattern)
            if args.num_formula_blocks == 0:
                chemical_formula = torch.zeros_like(chemical_formula)
            if args.num_lattice_blocks == 0:
                lattice_vector = torch.zeros_like(lattice_vector)
            if args.num_spacegroup_blocks == 0:
                spacegroup_vector = torch.zeros_like(spacegroup_vector)
            
            # load molecular id and formula
            molecular_id = test_loader.dataset.get_mp_id(test_loader.dataset.filepaths[idx].split('/')[-1])
            molecular_formula = test_loader.dataset.id_to_formulaStr[molecular_id]
            all_mpid_to_formula[molecular_id] = molecular_formula
            print(molecular_id, molecular_formula)
            # create results sub-folder
            curr_results_folder = os.path.join(args.results_folder, f'{molecular_id}_{molecular_formula}')
            os.makedirs(curr_results_folder, exist_ok=True)
            # plot diffraction pattern
            graph_diffraction_pattern(given_xrd_pattern[0], name=os.path.join(curr_results_folder, '{}_{}_xrd.png'.format(molecular_id, molecular_formula)), args=args)

            if args.plot_only_xrd:
                continue

            # check which elements contained
            if args.num_excluded_elems > 0:
                elements_as_list = chemical_formula.squeeze().tolist()
                ablated_formula = [atomic_num for atomic_num in range(len(elements_as_list)) \
                    if elements_as_list[atomic_num] > 0]
                print(ablated_formula)
                formula_file_name = os.path.join(curr_results_folder, f'{molecular_id}_{molecular_formula}_partial_elements_contained.json')
                with open(formula_file_name, 'w') as fout:
                    json.dump(ablated_formula, fout)

            # reshape pos
            assert pos.shape == (1, NUM_SAMPLES_PER_CRYSTAL, 3)
            pos = pos.squeeze()
            assert pos.shape == (NUM_SAMPLES_PER_CRYSTAL, 3)
            # reshape gt_charges
            assert gt_charges.shape == (1, NUM_SAMPLES_PER_CRYSTAL)
            gt_charges = gt_charges.squeeze()
            assert gt_charges.shape == (NUM_SAMPLES_PER_CRYSTAL, )

            # make sure loaded only one crystal
            assert given_xrd_pattern.shape[0] == 1
            assert given_xrd_pattern.shape[2] == 1024
            assert chemical_formula.shape == (1, 119)

            ## Plot GT crystal
            # create array coordinates from relative (x, y, z)
            grid_pos = torch.clone(pos)
            grid_pos[:,0] *= args.num_x
            grid_pos[:,1] *= args.num_y
            grid_pos[:,2] *= args.num_z
            grid_pos = grid_pos.int().cpu().numpy()
            # load raw spacegroup
            spacegroup_num = test_loader.dataset.id_to_spacegroup[molecular_id]
            # create gt crystal
            gt_crystal = np.zeros((args.num_x, args.num_y, args.num_z))
            gt_crystal[grid_pos[:,0], grid_pos[:,1], grid_pos[:,2]] = gt_charges.cpu().numpy()
            # plot gt crystal
            plot_charge_density(gt_crystal, range=(gt_crystal.min(), gt_crystal.max()), name=os.path.join(curr_results_folder, '{}_{}_gt.png'.format(molecular_id, molecular_formula)), 
                is_ground_truth=True, popup=args.display, multiple_camera_angles=args.multiple_camera_angles)

            # calculate embeddings first
            if args.num_channels > 0 and args.num_conv_blocks > 0:
                xrd_embedding_mean = model.diffraction_embedder_mean(given_xrd_pattern)
                xrd_embedding_std = model.diffraction_embedder_std(given_xrd_pattern)
                print(f'\tXRD mu: mean = {torch.round(xrd_embedding_mean.mean(), decimals=3)}, std = {torch.round(xrd_embedding_mean.std(), decimals=3)}')
                print(f'\tXRD sigma: mean = {torch.round(xrd_embedding_std.mean(), decimals=3)}, std = {torch.round(xrd_embedding_std.std(), decimals=3)}')                
            if args.num_formula_blocks > 0:
                formula_embedding_mean = model.formula_embedder_mean(chemical_formula)
                formula_embedding_std = model.formula_embedder_std(chemical_formula)
                print(f'\tFormula mu: mean = {torch.round(formula_embedding_mean.mean(), decimals=3)}, std = {torch.round(formula_embedding_mean.std(), decimals=3)}')
                print(f'\tFormula sigma: mean = {torch.round(formula_embedding_std.mean(), decimals=3)}, std = {torch.round(formula_embedding_std.std(), decimals=3)}')       
            # take a few random samples
            best_ssim = -1
            best_psnr = -1
            for trial_num in range(args.num_trials):
                if args.num_conv_blocks > 0:
                    # random sample XRD embedding
                    noise = torch.randn(xrd_embedding_mean.shape).to(xrd_embedding_std.device)
                    xrd_embedding = xrd_embedding_mean + noise * xrd_embedding_std
                    xrd_embedding = torch.cat([xrd_embedding for _ in range(args.batch_size)], dim=0) # just big enough for one batch
                else:
                    xrd_embedding = None
                if args.num_formula_blocks > 0:
                    # random sample formula embedding
                    noise = torch.randn(formula_embedding_mean.shape).to(formula_embedding_std.device)
                    formula_embedding = formula_embedding_mean + noise * formula_embedding_std
                    formula_embedding = torch.cat([formula_embedding for _ in range(args.batch_size)], dim=0) # just big enough for one batch
                else:
                    formula_embedding = None
                # predict!
                # batch size (# points tested at once) is different than number of crystals (one) loaded in each iteration of test loader
                pred_charges = list()
                for i in range(0, NUM_SAMPLES_PER_CRYSTAL, args.batch_size):
                    partial_pred_charges = model(diffraction_pattern=torch.zeros(1, device=given_xrd_pattern.device), 
                                            formula_vector=torch.zeros(1, device=chemical_formula.device), 
                                            lattice_vector=torch.zeros(1, device=lattice_vector.device), spacegroup_vector=torch.zeros(1, device=spacegroup_vector.device),
                                            position=pos[i:i+args.batch_size],
                                            diffraction_embedding=xrd_embedding[:NUM_SAMPLES_PER_CRYSTAL-i] if (args.num_channels > 0 and args.num_conv_blocks > 0) else None, 
                                            formula_embedding=formula_embedding[:NUM_SAMPLES_PER_CRYSTAL-i] if args.num_formula_blocks > 0 else None
                                        ).tolist()
                    pred_charges.extend(partial_pred_charges)
                assert len(pred_charges) == gt_charges.shape[0]
                assert len(pred_charges) == NUM_SAMPLES_PER_CRYSTAL
                pred_charges = np.array(pred_charges)

                # create pred crystal
                pred_crystal = np.zeros_like(gt_crystal)
                pred_crystal[grid_pos[:,0], grid_pos[:,1], grid_pos[:,2]] = pred_charges#.cpu().numpy()

                # generate vis
                plot_charge_density(pred_crystal, range=(gt_crystal.min(), gt_crystal.max()), name=os.path.join(curr_results_folder, '{}_{}_pred_{}.png'.format(molecular_id, molecular_formula, trial_num)), 
                    is_ground_truth=False, popup=args.display, multiple_camera_angles=args.multiple_camera_angles)

                # calculate metrics
                # ssim & psnr
                if args.test_all_rotations: # test all possible rotations of the prediction 
                    all_rotations = [curr_rot for curr_rot in rotations24(pred_crystal)] # calculate all rotations
                    curr_ssim = max([ssim(gt_crystal, curr_rotation, data_range=gt_crystal.max() - gt_crystal.min()) \
                                    for curr_rotation in all_rotations])
                    curr_psnr = max([psnr(image_true=gt_crystal, image_test=curr_rotation, data_range=gt_crystal.max() - gt_crystal.min()) \
                                    for curr_rotation in all_rotations])
                else: # just test the predicted rotation
                    curr_ssim = ssim(gt_crystal, pred_crystal, data_range=gt_crystal.max() - gt_crystal.min())
                    curr_psnr = psnr(image_true=gt_crystal, image_test=pred_crystal, data_range=gt_crystal.max() - gt_crystal.min())
                best_ssim = max(curr_ssim, best_ssim)
                best_psnr = max(curr_psnr, best_psnr)

            # log metrics (coarse-grained)
            # ssim
            print('\t\tssim: {}'.format(best_ssim))
            all_ssim.append(best_ssim)
            mpid_to_ssim[molecular_id] = best_ssim
            # psnr
            print('\t\tpsnr: {}'.format(best_psnr))
            all_psnr.append(best_psnr)
            mpid_to_psnr[molecular_id] = best_psnr

            # log metrics (fine-grained)
            # log by elements contained
            atomicNum_to_quantity = formula_to_composition(molecular_formula)
            for curr_atomicNum in atomicNum_to_quantity:
                ssim_by_elements[curr_atomicNum].append(best_ssim)
                psnr_by_elements[curr_atomicNum].append(best_psnr)
            # log number of atoms
            all_num_atoms.append(sum(atomicNum_to_quantity.values()))
            # log by spacegroup
            ssim_by_spacegroup[spacegroup_num].append(best_ssim)
            psnr_by_spacegroup[spacegroup_num].append(best_psnr)
    
    # calculate summary statistics (coarse-grained)
    avg_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)

    avg_psnr = np.mean(all_psnr)
    std_psnr = np.std(all_psnr)

    # calculate summary statistics (fine-grained)
    # do spacegroup
    dict_ssim_by_spacegroup = dict()
    dict_psnr_by_spacegroup = dict()
    for spacegroup_num in range(len(ssim_by_spacegroup)):
        if len(ssim_by_spacegroup[spacegroup_num]) > 0:
            dict_ssim_by_spacegroup[spacegroup_num] = (round(np.mean(ssim_by_spacegroup[spacegroup_num]), 4), 
                                          round(np.std(ssim_by_spacegroup[spacegroup_num]), 4))
            dict_psnr_by_spacegroup[spacegroup_num] = (round(np.mean(psnr_by_spacegroup[spacegroup_num]), 4), 
                                          round(np.std(psnr_by_spacegroup[spacegroup_num]), 4))

    results = {
        'avg ssim':round(avg_ssim, 4),
        'std ssim':round(std_ssim, 4),
        'avg psnr':round(avg_psnr, 2),
        'std psnr':round(std_psnr, 2),
        'crystals':all_mpid_to_formula,
        'ssim by crystal':mpid_to_ssim,
        'psnr by crystal':mpid_to_psnr,
        'ssim by spacegroup': dict_ssim_by_spacegroup,
        'psnr by spacegroup': dict_psnr_by_spacegroup
    }
    results.update(vars(args))

    curr_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    with open(os.path.join(args.results_folder, 'results_{}.json'.format(curr_time)), 'w') as fout:
        json.dump(results, fout, indent=4)
    
    print(json.dumps(results, indent=4))

    # graph summary statistics (fine-grained)
    graph_metric_by_elements(ssim_by_elements, filename=os.path.join(args.results_folder, 'ssim_by_atomicNum.pdf'), metric_name='SSIM')
    graph_metric_by_elements(psnr_by_elements, filename=os.path.join(args.results_folder, 'psnr_by_atomicNum.pdf'), metric_name='PSNR')
    graph_metric_by_numAtoms(all_num_atoms, all_ssim, filename=os.path.join(args.results_folder, 'ssim_by_numAtoms.pdf'), metric_name='SSIM')
    graph_metric_by_numAtoms(all_num_atoms, all_psnr, filename=os.path.join(args.results_folder, 'psnr_by_numAtoms.pdf'), metric_name='PSNR')

    return

def main():
    parser = argparse.ArgumentParser(description='Test the Charge Density Estimation Net')
    # Crystal representation parameters
    parser.add_argument('--num_x', type=int, default=5, metavar='NX',
                        help='Number of x samples per charge structure (default: 5)')
    parser.add_argument('--num_y', type=int, default=5, metavar='NY',
                        help='Number of y samples per charge structure (default: 5)')
    parser.add_argument('--num_z', type=int, default=5, metavar='NZ',
                        help='Number of z samples per charge structure (default: 5)')
    # Crystal data parameters
    parser.add_argument('--charge_data_dir', type=str, default='~/data/gabeguo/crystallography/charge_data_npy', metavar='D',
                        help='where is the charge data stored')
    parser.add_argument('--xrd_data_dir', type=str, default='~/data/gabeguo/crystallography/xrd_data_json/CuKa', metavar='D',
                        help='where is the xrd data stored')
    parser.add_argument('--mp_id_to_formula', type=str, default='~/data/gabeguo/crystallography/crystal_systems_all/Cubic_formulas.json',
                        help='JSON file with mapping of mp_id to chemical formula.')
    parser.add_argument('--mp_id_to_lattice', type=str, default='~/data/gabeguo/crystallography/crystal_systems_all/Cubic_lattice_vectors.json',
                        help='JSON file with mapping of mp_id to lattice vectors.')
    parser.add_argument('--mp_id_to_spacegroup', type=str, default='~/data/gabeguo/crystallography/crystal_systems_all/Cubic_space_groups.json',
                        help='JSON file with mapping of mp_id to spacegroup numbers.')
    parser.add_argument('--max_num_crystals', type=int, default=50,
                        help='how many train crystals to have in the dataset')
    # Data modality parameters
    parser.add_argument('--num_channels', type=int, default=2,
                        help='How many XRD channels to use (others will get masked out)')
    parser.add_argument('--ignore_elemental_ratios', action='store_true', default=False,
                        help='whether to express the chemical formula as just the elements, without considering ratios')
    parser.add_argument('--unstandardized_formula', action='store_true', default=False,
                        help='whether to express the n-hot elemental composition vector as 1s, or standardize it to sum to 1. ' + \
                            'this only matters if we ignore_elemental_ratios.')
    parser.add_argument('--num_excluded_elems', type=int, default=0,
                        help='number of elements to randomly drop out from chemical formula. ' + \
                            'this only matters if we ignore_elemental_ratios.')
    parser.add_argument('--use_mass_ratios', action='store_true',
                        help='whether to use mass ratios in elemental composition vector, or just have empirical formula')
    # Model weights
    parser.add_argument('--model_path', type=str, default='/data/therealgabeguo/crystallography/models/charge_density_net.pt', metavar='MP',
                        help='path to grab current model from') 
    # Model parameters
    parser.add_argument('--num_conv_blocks', type=int, default=4, \
                        help='how many convolutional blocks in each dense block (there are two dense blocks) (default: 4)')
    parser.add_argument('--num_regressor_blocks', type=int, default=3, \
                        help='how many blocks (e.g., four (fcs, BN, ReLU) without skip connection) to have in MLP regressor (default: 3)')
    parser.add_argument('--num_formula_blocks', type=int, default=3, \
                        help='how many fc layers to have in formula embedder (default: 3)')
    parser.add_argument('--num_lattice_blocks', type=int, default=3, \
                        help='how many fc layers to have in lattice embedder (default: 3)')
    parser.add_argument('--num_spacegroup_blocks', type=int, default=3, \
                        help='how many fc layers to have in spacegroup embedder (default: 3)')
    parser.add_argument('--num_freq', type=int, default=10, \
                        help='number of frequencies to use for Fourier features (default: 10)')
    parser.add_argument('--dropout_prob', type=float, default=0, \
                        help='probability of dropout in MLP regressor')
    # Data processing parameters
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='how many charges to process at once in testing')
    # Sampling parameters
    parser.add_argument('--num_trials', type=int, default=4,
                        help='how many times to try to generate a crystal')
    # Testing symmetry options
    parser.add_argument('--test_all_rotations', action='store_true', default=False, \
                        help='try all possible rotations of outputted molecule, and see which one best fits the ground truth')
    parser.add_argument('--output_com', action='store_true', default=False, \
                        help='output the center of mass of the ground truth crystal')
    # Output parameters    
    parser.add_argument('--results_folder', type=str, default='/data/therealgabeguo/crystallography/results/visualizations', \
                        help='Where to write the results')
    parser.add_argument('--multiple_camera_angles', action='store_true',
                        help='Whether to have three views of each crystal, or just the forward view in saved .png')
    parser.add_argument('--display', action='store_true', default=False,
                        help='whether to display crystal visualizations in browser')
    parser.add_argument('--plot_only_xrd', action='store_true',
                        help='Whether to plot only XRD, or also graph the molecule')
    # Miscellaneous
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    setattr(args, 'results_folder', os.path.join(args.results_folder, curr_time))
    os.makedirs(args.results_folder, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # test dataset
    test_dataset = CrystalDataset(args.charge_data_dir, args.xrd_data_dir, \
        chem_formula_path=args.mp_id_to_formula, lattice_path=args.mp_id_to_lattice, spacegroup_path=args.mp_id_to_spacegroup, \
        num_x_samples=args.num_x, num_y_samples=args.num_y, num_z_samples=args.num_z, \
        max_num_crystals=args.max_num_crystals, num_channels=args.num_channels, \
        ignore_elemental_ratios=args.ignore_elemental_ratios, \
        standardized_formula=not args.unstandardized_formula, num_excluded_elems=args.num_excluded_elems, \
        use_mass_ratios=args.use_mass_ratios, \
        train=False
    )

    global NUM_SAMPLES_PER_CRYSTAL
    NUM_SAMPLES_PER_CRYSTAL = args.num_x * args.num_y * args.num_z
    test_loader = torch.utils.data.DataLoader(test_dataset, 1)

    model = ChargeDensityRegressor(num_channels=args.num_channels, num_conv_blocks=args.num_conv_blocks, \
                num_formula_blocks=args.num_formula_blocks, \
                num_lattice_blocks=args.num_lattice_blocks, num_spacegroup_blocks=args.num_spacegroup_blocks, \
                num_regressor_blocks=args.num_regressor_blocks, \
                num_freq=args.num_freq, sigma=0, dropout_prob=args.dropout_prob).to(device)    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    evaluate(model, test_loader, device, args)

if __name__ == "__main__":
    main()
