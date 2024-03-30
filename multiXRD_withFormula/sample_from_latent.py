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

from chempy.util.parsing import formula_to_composition

from crystal_dataset import CrystalDataset
from crystal_mlp import ChargeDensityRegressor

import argparse
from tqdm import tqdm
from datetime import datetime

"""
Saves a visualization of the charge density map
"""
def plot_charge_density(density_map, the_range=(0, 1), name='charge_map.png', 
                        is_ground_truth=False, popup=False, multiple_camera_angles=False):
    X, Y, Z = np.mgrid[0:density_map.shape[0], 0:density_map.shape[1], 0:density_map.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=density_map.flatten(),
        isomin=the_range[0],
        isomax=the_range[1],
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

    # Cross section
    fig.data = []
    fig.layout = {}

    selected_z = 25
    cross_section = density_map[:,:,selected_z]
    X_cross = list()
    Y_cross = list()
    densities = list()
    for x in range(cross_section.shape[0]):
        for y in range(cross_section.shape[1]):
            X_cross.append(x)
            Y_cross.append(y)
            densities.append(cross_section[x, y])
    plt.figure(figsize=(8,8))
    plt.scatter(x=X_cross, y=Y_cross, c=densities, 
                marker='s', s=49, 
                vmin=density_map.min(), vmax=density_map.max(),
                linewidth=0, cmap='jet')
    plt.gca().set_aspect('equal')
    plt.savefig(f'{name_base}_crossSection{selected_z}.png')
    plt.close()
    
    print('\tvalues {}:'.format('ground truth' if is_ground_truth else 'predicted'), 
          '\n\t\tmean:', np.mean(density_map), '\n\t\tstd:', np.std(density_map))

    return

"""
Runs the model's predictions on every crystal in test_loader
"""
def evaluate(model, test_loader, device, args):
    model.eval()

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
            
            # create results sub-folder
            curr_results_folder = os.path.join(args.results_folder, 'random_samples')
            os.makedirs(curr_results_folder, exist_ok=True)

            if args.plot_only_xrd:
                continue

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

            # calculate embeddings first
            if args.num_channels > 0 and args.num_conv_blocks > 0:
                xrd_embedding_mean = model.diffraction_embedder_mean(given_xrd_pattern)
                xrd_embedding_std = model.diffraction_embedder_std(given_xrd_pattern)
                #print(f'\tXRD mu: mean = {torch.round(xrd_embedding_mean.mean(), decimals=3)}, std = {torch.round(xrd_embedding_mean.std(), decimals=3)}')
                #print(f'\tXRD sigma: mean = {torch.round(xrd_embedding_std.mean(), decimals=3)}, std = {torch.round(xrd_embedding_std.std(), decimals=3)}')                
            if args.num_formula_blocks > 0:
                formula_embedding_mean = model.formula_embedder_mean(chemical_formula)
                formula_embedding_std = model.formula_embedder_std(chemical_formula)
                #print(f'\tFormula mu: mean = {torch.round(formula_embedding_mean.mean(), decimals=3)}, std = {torch.round(formula_embedding_mean.std(), decimals=3)}')
                #print(f'\tFormula sigma: mean = {torch.round(formula_embedding_std.mean(), decimals=3)}, std = {torch.round(formula_embedding_std.std(), decimals=3)}')       
            
            xrd_embedding_mean = torch.zeros_like(xrd_embedding_mean)
            xrd_embedding_std = torch.ones_like(xrd_embedding_std)
            formula_embedding_mean = torch.zeros_like(formula_embedding_mean)
            formula_embedding_std = torch.ones_like(formula_embedding_std)

            # take a few random samples
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
                pred_crystal = np.zeros((args.num_x, args.num_y, args.num_z))
                pred_crystal[grid_pos[:,0], grid_pos[:,1], grid_pos[:,2]] = pred_charges#.cpu().numpy()

                

                # generate vis
                plot_charge_density(pred_crystal, the_range=(pred_crystal.min(), pred_crystal.max()), name=os.path.join(curr_results_folder, f'pred_{trial_num}.png'), 
                    is_ground_truth=True, popup=args.display, multiple_camera_angles=args.multiple_camera_angles)
            
            break

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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model has {total_params} parameters')

    evaluate(model, test_loader, device, args)

if __name__ == "__main__":
    main()
