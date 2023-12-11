import numpy as np
from pyrho.pgrid import PGrid
from pyrho.vis.scatter import get_scatter_plot

from pymatgen.io.vasp import Chgcar
from pyrho.charge_density import ChargeDensity

import plotly.graph_objects as go

import os
import sys
sys.path.append('./models')
from deeper_charge_density_mlp import *
from dense_mlp import *
import torch

import matplotlib.pyplot as plt
from matplotlib import mlab

from torchviz import make_dot, make_dot_from_trace

XRD_VECTOR_DIM = 1000

def plot_charge_histogram(density_map, standardize=True, n_bins=1000, range=(0, 1), name='diagram.png'):
    # density_map = normalize_map(density_map)
    # density_map = density_map.flatten().tolist()
    # density_map = np.array([x for x in density_map if x >= 0.2])
    # range = (density_map.min(), density_map.max())

    if standardize:
        density_map = normalize_map(density_map)
    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(density_map.flatten(), n_bins, histtype='step', cumulative=True, density=True)

    # tidy up the figure
    ax.grid(True)
    ax.set_title('CDF of charge density')
    ax.set_xticks(np.arange(range[0], range[1], (range[1] - range[0]) / 10))
    ax.set_xlabel('{}Charge density'.format('Standardized ' if standardize else ''))
    ax.set_ylabel('Cumulative frequency')

    plt.show()

    plt.savefig(name)

    return

def normalize_map(density_map):
    return (density_map - density_map.min()) / (density_map.max() - density_map.min())

def plot_charge_density(density_map, is_orig, standardize=True, range=(0, 1), name='charge_map.png'):
    X, Y, Z = np.mgrid[0:density_map.shape[0], 0:density_map.shape[1], 0:density_map.shape[2]]
    values = normalize_map(density_map) if standardize else density_map
    #print(values)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=range[0],
        isomax=range[1],
        opacity=0.8, # max opacity
        opacityscale=[(x, np.sqrt(x)) for x in np.linspace(start=0, stop=1, num=20)],
        surface_count=30,
        colorscale='rainbow'
        ))
    fig.write_image(name)
    fig.show()

    print('\tvalues {}:'.format('ground truth' if is_orig else 'predicted'), 
          '\n\t\tmean:', np.mean(values), '\n\t\tstd:', np.std(values))

    return

def make_backprop_graph(model, sampled_charge_vector):
    pred = model(sampled_charge_vector, torch.tensor([[0.5, 0.5, 0.5]]))
    make_dot(pred, params=dict(model.named_parameters())).render(directory='backprop_graph', filename='backprop_graph')
    return

# for item in [('train', 8195), ('train', 7788), ('train', 15363), ('train', 8033), ('train', 22266), ('train', 15886), \
#              ('train', 2861), ('val', 11918), ('val', 2863), ('val', 572)]:
#for item in [('train', 8195), ('train', 7788), ('train', 17501), ('train', 16848), ('val', 11918)]:
all_diffraction_embeddings = []
for item in [('val', 2863), ('val', 11918), ('val', 12005), ('val', 9987), ('test', 1920)]:
    print(item)
    """
    print('\t', charge_density.grid_shape)
    plt = get_scatter_plot(charge_density.pgrids['total'].grid_data, lat_mat=charge_density.pgrids['total'].lattice, plotter='plotly')
    plt.show()
    """

    # Original charge density
    new_charge_density = np.load('/home/gabeguo/data/crystallography/charge_data_npy/{}/CHGCAR_mp-{}.npy'.format(item[0], item[1]))
    
    plot_charge_histogram(new_charge_density, standardize=False, n_bins=200, range=(new_charge_density.min(), new_charge_density.max()),\
        name='/home/gabeguo/data/crystallography/outputs/visualizations/gt_histogram_{}.png'.format(item[1]))
    plot_charge_density(new_charge_density, is_orig=True, standardize=False, range=(new_charge_density.min(), new_charge_density.max()), \
        name='/home/gabeguo/data/crystallography/outputs/visualizations/gt_chargeMap_{}.png'.format(item[1]))

    # Predicted charge density
    model = DenseChargeDensityRegressor(num_blocks=8)#.to('cuda')
    #model.load_state_dict(torch.load('most_recent.pth'))
    model.load_state_dict(torch.load('/home/gabeguo/data/crystallography/models/charge_density_net_V45_8Layers_8192Batch.pt'))
    model.eval()

    #print('gaussian features std and mean:', torch.std_mean(model.position_embedder.frequencies))
    #print('state dict:', model.state_dict().keys())

    flattened = torch.flatten(torch.from_numpy(new_charge_density))
    sampled_charge_vector = torch.unsqueeze(torch.tensor([flattened[int(i / XRD_VECTOR_DIM * flattened.size(dim=0))] \
                            for i in range(XRD_VECTOR_DIM)]).to(dtype=torch.float32), dim=0)#.to('cuda')
    singular_diffraction_embedding = model.diffraction_embedder(sampled_charge_vector)
    random_embedding = torch.FloatTensor([[0 if i % 2 == 0 else 2 for i in range(512)]])

    diffraction_embeddings = list()
    positions = list()

    for x in np.arange(0, 1, 1 / new_charge_density.shape[0]):
        for y in np.arange(0, 1, 1 / new_charge_density.shape[1]):
            for z in np.arange(0, 1, 1 / new_charge_density.shape[2]):
                positions.append([x, y, z])
                diffraction_embeddings.append(singular_diffraction_embedding)

    diffraction_embeddings = torch.cat(diffraction_embeddings, dim=0)
    assert len(diffraction_embeddings.shape) == 2
    all_diffraction_embeddings.append(diffraction_embeddings)

    positions = torch.Tensor(positions).to(dtype=torch.float32)#.to('cuda')

    pred = model(sampled_charge_vector, positions, diffraction_embedding=diffraction_embeddings)

    make_backprop_graph(model, sampled_charge_vector)

    idx = 0
    pred_grid = np.zeros(new_charge_density.shape)
    for x in range(new_charge_density.shape[0]):
        for y in range(new_charge_density.shape[1]):
            for z in range(new_charge_density.shape[2]):
                pred_grid[x, y, z] = pred[idx]
                idx += 1
    plot_charge_histogram(pred_grid, standardize=False, n_bins=200, range=(new_charge_density.min(), new_charge_density.max()), \
        name='/home/gabeguo/data/crystallography/outputs/visualizations/pred_histogram_{}.png'.format(item[1]))
    plot_charge_density(pred_grid, is_orig=False, standardize=False, range=(new_charge_density.min(), new_charge_density.max()), \
        name='/home/gabeguo/data/crystallography/outputs/visualizations/pred_chargeMap_{}.png'.format(item[1]))

# # interpolation experiment
# for ratio in [0.2, 0.4, 0.6, 0.8]:
#     diffraction_embeddings = ratio * all_diffraction_embeddings[0] + (1 - ratio) * all_diffraction_embeddings[1]
#     pred = model(None, positions, diffraction_embedding=diffraction_embeddings)
#     idx = 0
#     pred_grid = np.zeros(new_charge_density.shape)
#     for x in range(new_charge_density.shape[0]):
#         for y in range(new_charge_density.shape[1]):
#             for z in range(new_charge_density.shape[2]):
#                 pred_grid[x, y, z] = pred[idx]
#                 idx += 1
#     plot_charge_histogram(pred_grid, standardize=False, n_bins=200, range=(new_charge_density.min(), new_charge_density.max()))
#     plot_charge_density(pred_grid, is_orig=False, standardize=False, range=(new_charge_density.min(), new_charge_density.max()))