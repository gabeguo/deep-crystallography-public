"""
Adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import os
from crystal_dataset import CrystalDataset, rotations24
import math
import datetime
import numpy as np
import random
from torchvision import transforms

import wandb
from tqdm import tqdm

import sys
sys.path.append('./models')
from crystal_mlp import ChargeDensityRegressor

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append('../pointnet/pointnet')
from model import PointNetCls

import torch
import math

import json

# Thanks ChatGPT! Verified via static analysis
def get_rotation_matrix(ax, ay, az):
    """
    Get the 3D rotation matrix for given angles ax, ay, and az.
    ax, ay, az: rotation angles around x, y, and z axes respectively in radians.
    """
    # Rotation matrices for each axis
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax), math.cos(ax)]
    ])

    Ry = torch.tensor([
        [math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)]
    ])

    Rz = torch.tensor([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az), math.cos(az), 0],
        [0, 0, 1]
    ])

    # Composite rotation: R = Rz * Ry * Rx
    R = torch.mm(Rz, torch.mm(Ry, Rx))

    return R

# Thanks ChatGPT! Verified via static analysis
def rotate_point_cloud(point_cloud, ax, ay, az):
    """
    Rotate a batch of point clouds around the x, y, and z axes.
    point_cloud: tensor of shape (B, N, 3)
    ax, ay, az: rotation angles in degrees around x, y, and z axes respectively.
    """
    B, N, space_dim = point_cloud.shape

    assert space_dim == 3

    # Convert angles to radians and get rotation matrix
    R = get_rotation_matrix(math.radians(ax), math.radians(ay), math.radians(az))

    # Expand the rotation matrix dimensions to multiply with point cloud
    R = R.unsqueeze(0).expand(B, -1, -1).to(point_cloud.device)

    # Rotate the point cloud
    rotated_point_cloud = torch.bmm(point_cloud, R.transpose(1, 2))

    assert rotated_point_cloud.shape == point_cloud.shape

    return rotated_point_cloud

def random_rotation_cube(point_cloud):
    """
    Rotate a batch of point clouds by a random multiple of 90 degrees about each axis
    point_cloud: tensor of shape (B, N, 3)
    """
    assert len(point_cloud.shape) == 3
    assert point_cloud.shape[2] == 3
    possible_angles = [0, 90, 180, 270]
    chosen_angles = np.random.choice(possible_angles, size=3)
    return rotate_point_cloud(point_cloud, chosen_angles[0], chosen_angles[1], chosen_angles[2])

# Thanks ChatGPT & https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        return
    def __call__(self, tensor):
        return F.relu(tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Thanks ChatGPT! Tested in Colab
class ShiftXRD(object):
    def __init__(self, max_shift=5):
        self.max_shift = max_shift
        return

    def __call__(self, x):
        assert len(x.shape) == 3
        assert self.max_shift < x.shape[2] - 1

        y = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # negative and positive shifts
                shift = np.random.randint(-self.max_shift, self.max_shift+1)         
                y[i, j] = torch.roll(x[i, j], shifts=shift, dims=0)
                # zero-pad       
                if shift >= 0:
                    y[i, j, :shift] = 0
                    assert torch.all(torch.eq(y[i, j, shift], x[i, j, 0]))
                else:
                    y[i, j, shift:] = 0
                    assert torch.all(torch.eq(x[i, j, -1], y[i, j, shift-1]))       
        return y

"""
Returns max_lambda * (-0.5 * cos(pi * curr_epoch / max_epoch) + 0.5) - verified
"""
def get_lambda(max_lambda, curr_epoch, max_epoch):
    factor = -0.5 * np.cos(np.pi * curr_epoch / max_epoch) + 0.5
    result = max_lambda * factor
    assert result >= 0
    assert result <= max_lambda
    if curr_epoch >= 1:
        assert result >= get_lambda(max_lambda, curr_epoch-1, max_epoch)
    return result

"""
Optimizes the discriminator for one step, with the assumption that we're being fed one batch worth of data
This is SEPARATE from generative optimization!
"""
def optimize_discriminator_one_step(classifier, optimizer, positions, gt_charges, pred_charges, args, show_results):
    # TODO: try just x, y, z?
    # TODO: try random rotation?

    # set to train mode, since we're optimizing now
    classifier.train()

    # check shapes
    assert len(positions.shape) == 3
    assert positions.shape[0] <= args.batch_size
    assert positions.shape[1] == args.num_x * args.num_y * args.num_z
    assert positions.shape[2] == 3

    # randomly rotate in one of 24 cubic orientations
    positions = random_rotation_cube(positions)

    assert gt_charges.shape == pred_charges.shape
    assert len(gt_charges.shape) == 2
    assert gt_charges.shape[0] == positions.shape[0]
    assert gt_charges.shape[1] == args.num_x * args.num_y * args.num_z

    # create "true" examples
    true_examples = torch.cat([positions, gt_charges.unsqueeze(dim=2)], dim=2)
    true_examples = torch.transpose(true_examples, 1, 2) # PointNet expects (batch, coordinates, number of points)
    assert true_examples.shape == (positions.shape[0], 4, args.num_x * args.num_y * args.num_z)
    #true_labels = torch.tensor([1 for _ in range(true_examples.shape[0])]).to(true_examples.device)
    true_labels = torch.bernoulli(torch.full(size=(true_examples.shape[0],), fill_value=0.9)).to(true_examples.device, dtype=torch.int64)

    # create "false" examples
    false_examples = torch.cat([positions, pred_charges.unsqueeze(dim=2)], dim=2)
    false_examples = torch.transpose(false_examples, 1, 2)
    assert false_examples.shape == true_examples.shape
    #false_labels = torch.tensor([0 for _ in range(false_examples.shape[0])]).to(false_examples.device)
    false_labels = torch.bernoulli(torch.full(size=(false_examples.shape[0],), fill_value=0.1)).to(false_examples.device, dtype=torch.int64)

    correct = 0
    # combine all examples
    for all_examples, target in  zip([true_examples, false_examples], [true_labels, false_labels]):
        # calculate predictions
        pred, trans, trans_feat = classifier(all_examples.detach()) # prevent from doing backprop through the generated examples
        # calculate discriminator loss
        loss = F.nll_loss(input=pred, target=target)

        # optimize DISCRIMINATOR
        optimizer.zero_grad() # get rid of any gradients that may have accumulated before
        loss.backward() # calculate new gradients
        optimizer.step() # update the discriminator with backprop
        optimizer.zero_grad() # get rid of any stored gradients - don't mess up any gradients for generator

        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum()

    # switch to eval mode, since we just need outputs
    classifier.eval()

    # calculate & return GENERATOR loss
    # Maximize log (D(G(z))), as suggested by original GAN paper - changed from negative discriminator loss
    pred_generated, _, _ = classifier(false_examples) # only need to calculate for generated samples
    assert len(pred_generated.shape) == 2
    assert pred_generated.shape[1] == 2
    # train generator to fool the discriminator, so flip the labels
    generative_loss = F.nll_loss(input=pred_generated, 
        target=torch.tensor([1 for _ in range(true_examples.shape[0])]).to(true_examples.device)) # get log-prob that discriminator predicted true for generated samples

    # show accuracy
    if show_results:
        print('\ttrain loss discriminator: %f accuracy: %f' % (loss.item(), correct.item() / float(2 * all_examples.shape[0])))
        print(f'\ttrain loss generator: {generative_loss.item()}')

    # get perceptual similarity loss
    crystal_features_gt, _, _ = classifier.feat(true_examples) # extract feature embedding for real crystal
    crystal_features_pred, _, _ = classifier.feat(false_examples) # extract feature embedding for predicted crystal
    assert crystal_features_gt.shape == crystal_features_pred.shape
    assert crystal_features_gt.shape[0] == pred_generated.shape[0]
    assert crystal_features_gt.shape[1] == 1024

    deep_feature_similarities = F.cosine_similarity(crystal_features_gt, crystal_features_pred)
    assert deep_feature_similarities.shape == (crystal_features_gt.shape[0],)
    # NEGATE cosine, because we want to maximize cosine, which is same as minimizing -cos
    cosine_loss = 1 - torch.mean(deep_feature_similarities)
    assert cosine_loss >= 0 and cosine_loss <= 1

    return generative_loss, cosine_loss, correct

def train(args, model, device, train_loader, optimizer, loss_fn, epoch,
          discriminator, discriminator_optimizer):
    print('Train Epoch: {}'.format(epoch))
    model.train()
    avg_train_loss = 0
    total_discriminated_correct = 0

    xrd_transform = transforms.Compose([
        ShiftXRD(max_shift=args.xrd_shift), # shift first, so that we have chance to add noise to 0
        AddGaussianNoise(mean=0., std=args.xrd_noise)
    ])
    formula_transform = AddGaussianNoise(0., args.formula_noise)

    NUM_SAMPLES_PER_CRYSTAL_TRAIN = args.num_x * args.num_y * args.num_z

    for batch_idx, the_tuple in enumerate(tqdm(train_loader)):
        given_xrd_pattern, chem_formula, lattice_params, spacegroup, pos, charge_at_pos = the_tuple
        
        given_xrd_pattern = given_xrd_pattern.to(device)
        chem_formula = chem_formula.to(device)
        lattice_params = lattice_params.to(device)
        spacegroup = spacegroup.to(device)
        pos = pos.to(device)
        charge_at_pos = charge_at_pos.to(device)

        # data augmentation
        if args.augmentation:
            given_xrd_pattern = xrd_transform(given_xrd_pattern)
            chem_formula = formula_transform(chem_formula)

        # eliminate data if needed
        if args.num_channels == 0 or args.num_conv_blocks == 0:
            given_xrd_pattern = torch.zeros_like(given_xrd_pattern)
        if args.num_formula_blocks == 0:
            chem_formula = torch.zeros_like(chem_formula)
        if args.num_lattice_blocks == 0:
            lattice_params = torch.zeros_like(lattice_params)
        if args.num_spacegroup_blocks == 0:
            spacegroup = torch.zeros_like(spacegroup)

        # check to make sure the shapes are what we think they are
        assert given_xrd_pattern.shape[2] == 1024
        curr_batch_size = given_xrd_pattern.shape[0]
        assert pos.shape == (curr_batch_size, NUM_SAMPLES_PER_CRYSTAL_TRAIN, 3) # batched by crystal
        assert charge_at_pos.shape == (curr_batch_size, NUM_SAMPLES_PER_CRYSTAL_TRAIN) # batched by crystal

        # reshape so we can feed into DL model
        super_batch_pos = pos.reshape(-1, 3) # "flattened"
        super_batch_charge = charge_at_pos.reshape(-1) # "flattened"
        assert super_batch_pos.shape == (curr_batch_size * NUM_SAMPLES_PER_CRYSTAL_TRAIN, 3)
        assert super_batch_charge.shape == (curr_batch_size * NUM_SAMPLES_PER_CRYSTAL_TRAIN, )

        # Precompute embeddings
        
        # XRD
        if args.num_channels > 0 and args.num_conv_blocks > 0: # we need them
            xrd_thetas = list() # [0]->mean, [1]->std
            for curr_xrd_embedder in [model.diffraction_embedder_mean, model.diffraction_embedder_std]:
                # get embeddings
                xrd_embeddings = curr_xrd_embedder(given_xrd_pattern)
                assert len(xrd_embeddings) == curr_batch_size
                xrd_thetas.append(xrd_embeddings)
            # variational approach
            noise = torch.randn(xrd_thetas[0].shape).to(xrd_thetas[1].get_device())
            xrd_embeddings = xrd_thetas[0] + noise * xrd_thetas[1]
            # get KLD loss
            xrd_kld = -args.kl_weight * torch.mean(1 + torch.log(xrd_thetas[1]**2 + 1e-4) - xrd_thetas[0]**2 - xrd_thetas[1]**2)
            # repeat embeddings
            xrd_embeddings = torch.repeat_interleave(xrd_embeddings, NUM_SAMPLES_PER_CRYSTAL_TRAIN, dim=0)
            # check that shape is correct
            assert xrd_embeddings.shape == (curr_batch_size * NUM_SAMPLES_PER_CRYSTAL_TRAIN, 512)
            # check that we repeated them correctly
            assert torch.equal(xrd_embeddings[0], xrd_embeddings[NUM_SAMPLES_PER_CRYSTAL_TRAIN-1])
            # store this parameter
        else: # not using this info
            xrd_embeddings = None #torch.zeros(curr_batch_size * NUM_SAMPLES_PER_CRYSTAL_TRAIN, 512).to(given_xrd_pattern.get_device())
            xrd_kld = 0

        # Formula
        if args.num_formula_blocks > 0: # we need them
            formula_thetas = list() # [0]->mean, [1]->std
            for curr_formula_embedder in [model.formula_embedder_mean, model.formula_embedder_std]:
                # get embeddings
                formula_embeddings = curr_formula_embedder(chem_formula)
                assert len(formula_embeddings) == curr_batch_size
                # store this parameter
                formula_thetas.append(formula_embeddings)
            # variational approach
            noise = torch.randn(formula_thetas[1].shape).to(formula_thetas[0].get_device())
            formula_embeddings = formula_thetas[0] + noise * formula_thetas[1]
            # get KLD loss
            formula_kld = -args.kl_weight * torch.mean(1 + torch.log(formula_thetas[1]**2 + 1e-4) - formula_thetas[0]**2 - formula_thetas[1]**2)
            # repeat embeddings
            formula_embeddings = torch.repeat_interleave(formula_embeddings, NUM_SAMPLES_PER_CRYSTAL_TRAIN, dim=0)
            # check that shape is correct
            assert formula_embeddings.shape == (curr_batch_size * NUM_SAMPLES_PER_CRYSTAL_TRAIN, 512)
            # check that we repeated them correctly
            assert torch.equal(formula_embeddings[0], formula_embeddings[1])
        else: # not using this info
            formula_embeddings = None #torch.zeros(curr_batch_size * NUM_SAMPLES_PER_CRYSTAL_TRAIN, 512).to(chem_formula.get_device())
            formula_kld = 0

        # predict the charges
        pred_charges = model(diffraction_pattern=given_xrd_pattern, formula_vector=chem_formula, 
                lattice_vector=lattice_params, spacegroup_vector=spacegroup, 
                position=super_batch_pos, # use batched batched positions (2D; all crystals together)
                diffraction_embedding=xrd_embeddings, formula_embedding=formula_embeddings)  # use precomputed embeddings
        # check size
        assert pred_charges.shape == super_batch_charge.shape
        # calculate reconstruction loss
        reconstruction_loss = loss_fn(pred_charges, super_batch_charge)

        # reshape into crystals
        pred_charges_by_crystal = pred_charges.reshape(charge_at_pos.shape)

        # get loss
        if args.lambda_sim == 0 and args.lambda_adv == 0: # don't do unnecessary computation
            adversarial_loss, similarity_loss, discriminated_correct = 0, 0, 0
            loss = reconstruction_loss
        else: # we're using adversarial losses
            adversarial_loss, similarity_loss, discriminated_correct = optimize_discriminator_one_step(classifier=discriminator, 
                                    optimizer=discriminator_optimizer, 
                                    positions=pos, gt_charges=charge_at_pos, pred_charges=pred_charges_by_crystal, 
                                    args=args, show_results = (batch_idx % args.log_interval == 0))

            # combined loss
            loss = reconstruction_loss \
                + get_lambda(max_lambda=args.lambda_sim, curr_epoch=epoch, max_epoch=args.epochs) * similarity_loss \
                + get_lambda(max_lambda=args.lambda_adv, curr_epoch=epoch, max_epoch=args.epochs) * adversarial_loss

        # # Add KLD
        loss += (xrd_kld + formula_kld)

        # optimize
        # calculate gradients w.r.t. combined loss
        optimizer.zero_grad() # clear gradients before, to be super safe
        loss.backward() # can clear computation graph now (last time)
        optimizer.step() # backprop
        # IMPORTANT - zero_grad is below the logging code

        # calculate loss
        avg_train_loss += loss.item() / len(train_loader)
        total_discriminated_correct += discriminated_correct
        if batch_idx % args.log_interval == 0:
            print('\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(given_xrd_pattern), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            avg_final_linear_gradient = torch.mean(model.lastfc.weight.grad)
            std_final_linear_gradient = torch.std(model.lastfc.weight.grad)
            print('\t\tgradient at end: mean = {:.2E} std = {:.2E}'.format(avg_final_linear_gradient, std_final_linear_gradient))
            avg_final_linear_weight = torch.mean(model.lastfc.weight)
            std_final_linear_weight = torch.std(model.lastfc.weight)
            print('\t\tweight at end: mean = {:.2E} std = {:.2E}'.format(avg_final_linear_weight, std_final_linear_weight))    
            avg_prediction = torch.mean(pred_charges)
            std_prediction = torch.std(pred_charges)
            print('\t\tprediction: mean = {:.4f} std = {:.4f}'.format(avg_prediction, std_prediction))
            avg_value = torch.mean(charge_at_pos)
            std_value = torch.std(charge_at_pos)
            print('\t\tground truth: mean = {:.4f} std = {:.4f}'.format(avg_value, std_value))
            dist_index = torch.randint(low=0, high=512, size=(1,)).item()
            print('\t\t\tXRD latent: mean = {:.4f} std = {:.4f}'.format(xrd_thetas[0][0, dist_index], xrd_thetas[1][0, dist_index]))
            if args.num_formula_blocks > 0:
                print('\t\t\tFormula latent: mean = {:.4f} std = {:.4f}'.format(formula_thetas[0][0, dist_index], formula_thetas[1][0, dist_index]))
            print('\t\t\tKLD XRD: {:.3E}'.format(xrd_kld))
            print('\t\t\tKLD Formula: {:.3E}'.format(formula_kld))
            
            if args.dry_run:
                break
        
        # IMPORTANT - zero grad
        optimizer.zero_grad() # clear gradients, so that we're clear for next time and for the discriminator

    print('\taverage train loss = {:.6f}'.format(avg_train_loss))
    discriminator_accuracy = total_discriminated_correct / (2 * len(train_loader.dataset))
    print('\tdiscriminator accuracy = {:.6f}'.format(discriminator_accuracy))
    return avg_train_loss, discriminator_accuracy


AVG_SSIM = 'avg ssim'
STD_SSIM = 'std ssim'
AVG_PSNR = 'avg psnr'
STD_PSNR = 'std psnr'
"""
Runs the model's predictions on every crystal in test_loader
"""
def evaluate(model, test_loader, device, args):
    model.eval()

    # general results
    all_ssim = list()
    all_psnr = list()

    NUM_SAMPLES_PER_CRYSTAL_TEST = args.num_x_val * args.num_y_val * args.num_z_val

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

            # reshape pos
            assert pos.shape == (1, NUM_SAMPLES_PER_CRYSTAL_TEST, 3)
            pos = pos.squeeze()
            assert pos.shape == (NUM_SAMPLES_PER_CRYSTAL_TEST, 3)
            # reshape gt_charges
            assert gt_charges.shape == (1, NUM_SAMPLES_PER_CRYSTAL_TEST)
            gt_charges = gt_charges.squeeze()
            assert gt_charges.shape == (NUM_SAMPLES_PER_CRYSTAL_TEST, )

            # make sure loaded only one crystal
            assert given_xrd_pattern.shape[0] == 1
            assert given_xrd_pattern.shape[2] == 1024
            assert chemical_formula.shape == (1, 119)

            # batch size is different than number of crystals loaded in each iteration of test loader
            # batch size is how many points the network tests at once
            # the test loader will load exactly one crystal per iteration
            if args.num_channels > 0 and args.num_conv_blocks > 0: # only if we need
                xe_mean = model.diffraction_embedder_mean(given_xrd_pattern)
                xe_std = model.diffraction_embedder_std(given_xrd_pattern)
            # else: # fill with 0
            #     xrd_embedding = None #torch.zeros(given_xrd_pattern.shape[0], 512).to(given_xrd_pattern.get_device())
            if args.num_formula_blocks > 0: # only if we need
                fe_mean = model.formula_embedder_mean(chemical_formula)
                fe_std = model.formula_embedder_std(chemical_formula)
            # else: # fill with 0
            #     formula_embedding = None #torch.zeros(chemical_formula.shape[0], 512).to(chemical_formula.get_device())

            # sample from latent vector distribution multiple times, and take best result (since it's variational)
            best_ssim = -1
            best_psnr = -1
            for trial in range(args.num_val_trials):
                # XRD sample
                if args.num_conv_blocks > 0:
                    xrd_noise = torch.normal(mean=0, std=1, size=xe_mean.shape).to(xe_std.get_device())
                    xrd_embedding = xe_mean + xrd_noise * xe_std
                    xrd_embedding = torch.cat([xrd_embedding for _ in range(args.test_batch_size)], dim=0) # just big enough for one batch
                else:
                    xrd_embedding = None
                # Formula sample
                if args.num_formula_blocks > 0:
                    formula_noise = torch.normal(mean=0, std=1, size=fe_mean.shape).to(fe_std.get_device())
                    formula_embedding = fe_mean + formula_noise * fe_std
                    formula_embedding = torch.cat([formula_embedding for _ in range(args.test_batch_size)], dim=0) # just big enough for one batch
                else:
                    formula_embedding = None

                # this code handles large batches
                pred_charges = list()
                for i in range(0, NUM_SAMPLES_PER_CRYSTAL_TEST, args.test_batch_size):
                    partial_pred_charges = model(given_xrd_pattern[i:i+args.test_batch_size], 
                                            formula_vector=chemical_formula[i:i+args.test_batch_size], 
                                            lattice_vector=lattice_vector[i:i+args.test_batch_size], spacegroup_vector=spacegroup_vector[i:i+args.test_batch_size],
                                            position=pos[i:i+args.test_batch_size],
                                            diffraction_embedding=xrd_embedding[:NUM_SAMPLES_PER_CRYSTAL_TEST-i] if (args.num_channels > 0 and args.num_conv_blocks > 0) else None, 
                                            formula_embedding=formula_embedding[:NUM_SAMPLES_PER_CRYSTAL_TEST-i] if args.num_formula_blocks > 0 else None
                                        ).tolist()
                    pred_charges.extend(partial_pred_charges)
                assert len(pred_charges) == gt_charges.shape[0]
                assert len(pred_charges) == NUM_SAMPLES_PER_CRYSTAL_TEST
                pred_charges = np.array(pred_charges)

                # create array coordinates from relative (x, y, z)
                grid_pos = torch.clone(pos)
                grid_pos[:,0] *= args.num_x_val # use validation size, NOT train size
                grid_pos[:,1] *= args.num_y_val
                grid_pos[:,2] *= args.num_z_val
                grid_pos = grid_pos.int().cpu().numpy()
                assert grid_pos.max() == max(args.num_x_val, args.num_y_val, args.num_z_val) - 1

                # create gt crystal
                gt_crystal = np.zeros((args.num_x_val, args.num_y_val, args.num_z_val))
                gt_crystal[grid_pos[:,0], grid_pos[:,1], grid_pos[:,2]] = gt_charges.cpu().numpy()
                # create pred crystal
                pred_crystal = np.zeros_like(gt_crystal)
                pred_crystal[grid_pos[:,0], grid_pos[:,1], grid_pos[:,2]] = pred_charges
                assert gt_crystal.shape == (args.num_x_val, args.num_y_val, args.num_z_val)
                assert gt_crystal.shape == pred_crystal.shape

                # calculate metrics
                # ssim & psnr @ all possible rotations of the crystal
                all_rotations = [curr_rot for curr_rot in rotations24(pred_crystal)] # calculate all rotations
                curr_top_ssim = max([ssim(gt_crystal, curr_rotation, data_range=gt_crystal.max() - gt_crystal.min()) \
                                for curr_rotation in all_rotations])
                curr_top_psnr = max([psnr(image_true=gt_crystal, image_test=curr_rotation, data_range=gt_crystal.max() - gt_crystal.min()) \
                                for curr_rotation in all_rotations])

                best_ssim = max(best_ssim, curr_top_ssim)
                best_psnr = max(best_psnr, curr_top_psnr)
            # log best metrics over all n trials
            all_ssim.append(best_ssim)
            all_psnr.append(best_psnr)
            
    assert len(all_ssim) == idx + 1
    assert len(all_psnr) == len(all_ssim)

    # calculate summary statistics (coarse-grained)
    avg_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)

    avg_psnr = np.mean(all_psnr)
    std_psnr = np.std(all_psnr)

    results = {
        AVG_SSIM:round(avg_ssim, 4),
        STD_SSIM:round(std_ssim, 4),
        AVG_PSNR:round(avg_psnr, 2),
        STD_PSNR:round(std_psnr, 2),
    }

    return results

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train the Charge Density Estimation Net')
    # optimization params
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='Number of distinct crystals to load on each training iteration (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=8192, metavar='N',
                        help='Number of charges to process at once during testing (default: 8192) ' + \
                        '- note this has DIFFERENT meaning than train batch size, as we do one crystal at a time in testing')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    # optimization params - generator
    parser.add_argument('--kl_weight', type=float, default=0.5,
                        help='how much to weight KLD loss')
    parser.add_argument('--mse', action='store_true', default=False,
                        help='whether to use MSE Loss (otherwise, use L1 loss)')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='Whether to use Adam (otherwise, use SGD with momentum)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='how much momentum to use for SGD')
    parser.add_argument('--weight_decay', type=float, default=0, 
                        help='how much weight decay to use')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warm_restart', action='store_true', default=False,
                        help='Whether to use warm restarts with cosine annealing (otherwise, does StepLR)')
    parser.add_argument('--lr_step', type=int, default=1, metavar='LR_Step',
                        help='learning rate step interval, or number of epochs between warm restarts (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--min_lr', type=float, default=1e-4,
                        help='Minimum learning rate when using cosine annealing (default: 1e-4)')
    # optimization params - discriminator
    parser.add_argument('--discrim_adam', action='store_true', default=False, 
                        help='Whether to use Adam (otherwise, use SGD with momentum) for the discriminator')
    parser.add_argument('--discrim_momentum', type=float, default=0.9, 
                        help='how much momentum to use for SGD for the discriminator')
    parser.add_argument('--discrim_weight_decay', type=float, default=0, 
                        help='how much weight decay to use for the discriminator')
    parser.add_argument('--discrim_lr', type=float, default=1e-3,
                        help='learning rate for the discriminator (default: 1e-3)')
    parser.add_argument('--discrim_warm_restart', action='store_true', default=False, 
                        help='Whether to use warm restarts with cosine annealing (otherwise, does StepLR) for the discriminator')
    parser.add_argument('--discrim_lr_step', type=int, default=1, 
                        help='Number of epochs for the first restart for the discriminator scheduler.')
    parser.add_argument('--discrim_gamma', type=float, default=0.7,
                        help='An integer factor to scale max learning rate by after a restart for the discriminator scheduler or multiplicative factor of learning rate decay.')
    parser.add_argument('--discrim_min_lr', type=float, default=1e-4, 
                        help='Minimum learning rate for the discriminator (default: 1e-4)')
    # optimization params - combined
    parser.add_argument('--lambda_adv', type=float, default=1e-3,
                        help='Weight to give adversarial loss; assume L1 loss has weight 1 (default: 1e-3)')
    parser.add_argument('--lambda_sim', type=float, default=0,
                        help='Weight to give deep perceptual similarity loss (default: 0)')
    # crystal sampling params
    parser.add_argument('--num_x', type=int, default=5, metavar='NX',
                        help='Number of x samples per charge structure (default: 5)')
    parser.add_argument('--num_y', type=int, default=5, metavar='NY',
                        help='Number of y samples per charge structure (default: 5)')
    parser.add_argument('--num_z', type=int, default=5, metavar='NZ',
                        help='Number of z samples per charge structure (default: 5)')
    parser.add_argument('--num_x_val', type=int, default=5,
                        help='Number of x samples per charge structure in validation set (default: 5)')
    parser.add_argument('--num_y_val', type=int, default=5,
                        help='Number of y samples per charge structure in validation set (default: 5)')
    parser.add_argument('--num_z_val', type=int, default=5,
                        help='Number of z samples per charge structure in validation set (default: 5)')
    # dataset params
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
    parser.add_argument('--max_num_val_crystals', type=int, default=20,
                        help='how many val crystals to have in the dataset')
    parser.add_argument('--same_train_val', action='store_true',
                        help='whether to use the same training and val set (only for dev purposes)')   
    # extra dataset params
    parser.add_argument('--alt_train_charge_folder', nargs='*', type=str, default=None,
                        help='If set, we will use these as extra folders to take charge training data from (train not appended to path). ' + \
                            'Note that alt_train_xrd_folder must also be set for this to work. ' + \
                            'Corresponding items must be in same order (and same total number) as alt_train_xrd_folder.')
    parser.add_argument('--alt_train_xrd_folder', nargs='*', type=str, default=None,
                        help='If set, we will use these as extra folders to take xrd training data from (train not appended to path). ' + \
                            'Note that alt_train_charge_folder must also be set for this to work. ' + \
                            'Corresponding items must be in same order (and same total number) as alt_train_charge_folder.')
    parser.add_argument('--alt_mp_id_to_formula', type=str, default=None,
                        help='JSON file with mapping of mp_id to chemical formula for extra train dataset. Default: None, use mp_id_to_formula.')
    parser.add_argument('--alt_mp_id_to_lattice', type=str, default=None,
                        help='JSON file with mapping of mp_id to lattice vectors for extra train dataset. Default: None, use mp_id_to_lattice.')
    parser.add_argument('--alt_mp_id_to_spacegroup', type=str, default=None,
                        help='JSON file with mapping of mp_id to spacegroup numbers for extra train dataset. Default: None, use mp_id_to_spacegroup.')
    # data modality params
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
    
    # data augmentation params
    parser.add_argument('--augmentation', action='store_true', default=False,
                        help='whether to do data augmentation')
    parser.add_argument('--xrd_noise', type=float, default=1e-4,
                        help='How much noise (sigma in Gaussian) to add to XRD tensor')
    parser.add_argument('--formula_noise', type=float, default=1e-6,
                        help='How much noise (sigma in Gaussian) to add to formula tensor')
    parser.add_argument('--xrd_shift', type=int, default=5,
                        help='Maximum amount to randomly shift XRD by')
    # model params
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
    parser.add_argument('--freq_sigma', type=float, default=1, \
                        help='Sigma of randomly drawn Fourier coordinate frequencies')
    parser.add_argument('--dropout_prob', type=float, default=0, \
                        help='probability of dropout in MLP regressor')
    # val params
    parser.add_argument('--num_val_trials', type=int, default=3, \
                        help='number of samples to take from latent variable distribution in evaluation')
    # operational params
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=500, metavar='LI',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_log_interval', type=int, default=150, metavar='LI',
                        help='how many batches to wait before logging validation status')
    parser.add_argument('--patience', type=int, default=5, metavar='P',
                        help='how long to train model before early stopping, if no improvement')
    parser.add_argument('--model_path', type=str, default='/data/therealgabeguo/crystallography/models/charge_density_net.pt', metavar='MP',
                        help='path to save the current model at')
    parser.add_argument('--results_folder', type=str, default='/data/therealgabeguo/crystallography/run_logs', \
                        help='Where to write the results')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='What checkpoint to use')
    parser.add_argument('--wandb_project', type=str, default=None, \
                        help='database name for wandb')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.wandb_project:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=args.wandb_project,
            name=args.model_path.split('/')[-1],
            # Track hyperparameters and run metadata
            config=vars(args)
        )

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('train loop loading...')


    # train dataset
    train_dataset = CrystalDataset(os.path.join(args.charge_data_dir, 'train'), os.path.join(args.xrd_data_dir, 'train'), 
        chem_formula_path=args.mp_id_to_formula, lattice_path=args.mp_id_to_lattice, spacegroup_path=args.mp_id_to_spacegroup, 
        num_x_samples=args.num_x, num_y_samples=args.num_y, num_z_samples=args.num_z, 
        max_num_crystals=args.max_num_crystals, num_channels=args.num_channels, 
        ignore_elemental_ratios=args.ignore_elemental_ratios, 
        standardized_formula=not args.unstandardized_formula, num_excluded_elems=args.num_excluded_elems,
        use_mass_ratios=args.use_mass_ratios,
        train=True
    )
    # extra train data (separate source)
    if (args.alt_train_charge_folder is not None) and (args.alt_train_xrd_folder is not None): 
        assert len(args.alt_train_charge_folder) == len(args.alt_train_xrd_folder)
        # add all the datasets
        curr_chem_formula_path = args.alt_mp_id_to_formula if args.alt_mp_id_to_formula \
            else args.mp_id_to_formula
        curr_lattice_path = args.alt_mp_id_to_lattice if args.alt_mp_id_to_lattice \
            else args.mp_id_to_lattice
        curr_spacegroup_path = args.alt_mp_id_to_spacegroup if args.alt_mp_id_to_spacegroup \
            else args.mp_id_to_spacegroup
        all_datasets = [train_dataset]
        for curr_charge_folder, curr_xrd_folder in zip(args.alt_train_charge_folder, args.alt_train_xrd_folder):
            extra_train_dataset = CrystalDataset(curr_charge_folder, curr_xrd_folder, 
                chem_formula_path=curr_chem_formula_path, lattice_path=curr_lattice_path, spacegroup_path=curr_spacegroup_path, 
                num_x_samples=args.num_x, num_y_samples=args.num_y, num_z_samples=args.num_z, 
                max_num_crystals=args.max_num_crystals, num_channels=args.num_channels, 
                ignore_elemental_ratios=args.ignore_elemental_ratios, 
                standardized_formula=not args.unstandardized_formula, num_excluded_elems=args.num_excluded_elems,
                use_mass_ratios=args.use_mass_ratios,
                train=True
            )
            all_datasets.append(extra_train_dataset)
        train_dataset = torch.utils.data.ConcatDataset(all_datasets)

    # val dataset
    val_dataset = CrystalDataset(os.path.join(args.charge_data_dir, 'train' if args.same_train_val else 'val'),
        os.path.join(args.xrd_data_dir, 'train' if args.same_train_val else 'val'),
        chem_formula_path=args.mp_id_to_formula, lattice_path=args.mp_id_to_lattice, spacegroup_path=args.mp_id_to_spacegroup,
        num_x_samples=args.num_x_val, num_y_samples=args.num_y_val, num_z_samples=args.num_z_val, # very important to set this to number of val samples
        max_num_crystals=args.max_num_val_crystals, num_channels=args.num_channels,
        ignore_elemental_ratios=args.ignore_elemental_ratios, 
        standardized_formula=not args.unstandardized_formula, num_excluded_elems=args.num_excluded_elems,
        use_mass_ratios=args.use_mass_ratios,
        train=False # TESTING now
    )

    train_kwargs = {'batch_size': args.batch_size}
     # test loader will load one crystal per iteration
    test_kwargs = {'batch_size': 1}
    if use_cuda:
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **train_kwargs) # shuffle training crystal order
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **test_kwargs)

    print('dataset loaded')

    # create generative model
    model = ChargeDensityRegressor(num_channels=args.num_channels, num_conv_blocks=args.num_conv_blocks, \
            num_formula_blocks=args.num_formula_blocks, num_lattice_blocks=args.num_lattice_blocks, num_spacegroup_blocks=args.num_spacegroup_blocks, \
            num_regressor_blocks=args.num_regressor_blocks, \
            num_freq=args.num_freq, sigma=args.freq_sigma, dropout_prob=args.dropout_prob).to(device)    
    # generative optimizer
    if args.adam:
        print('adam - generative')
        optimizer = optim.Adam(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('sgd - generative')
        optimizer = optim.SGD(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    # generative loss
    if args.mse:
        print('mse loss')
        train_loss_fn = torch.nn.MSELoss(reduction='mean')
    else:
        print('l1 loss')
        train_loss_fn = torch.nn.L1Loss(reduction='mean')      

    # create discriminative model
    discriminator = PointNetCls(k=2).to(device) # binary classification - real or fake
    # discriminative optimizer
    if args.discrim_adam:
        print('adam - discriminator')
        discriminator_optimizer = optim.Adam(discriminator.parameters(), # for DISCRIMINATOR!
                                lr=args.discrim_lr, weight_decay=args.discrim_weight_decay)
    else:
        print('sgd - discriminator')
        discriminator_optimizer = optim.SGD(discriminator.parameters(), # for DISCRIMINATOR, not generator!
                               lr=args.discrim_lr, weight_decay=args.discrim_weight_decay, momentum=args.discrim_momentum)
    # create path for model
    if not os.path.exists(args.model_path):
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    num_iterations_no_improvement = 0
    # scheduler - generator
    if args.warm_restart:
        t_mult = max(1, int(args.gamma))
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                        T_0 = args.lr_step,# Number of iterations for the first restart
                                        T_mult = t_mult, # An integer factor to scale max learning rate by after a restart
                                        eta_min = args.min_lr) # Min learning rate
        print('cosine annealing - generator')
    else:
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
        print('step lr - generator')
    # scheduler - discriminator
    if args.discrim_warm_restart:
        t_mult = max(1, int(args.discrim_gamma))
        discriminator_scheduler = CosineAnnealingWarmRestarts(discriminator_optimizer, # warm restarts for DISCRIMINATOR!
                                        T_0 = args.discrim_lr_step,# Number of iterations for the first restart
                                        T_mult = t_mult, # An integer factor to scale max learning rate by after a restart
                                        eta_min = args.discrim_min_lr) # Min learning rate
        print('cosine annealing - discriminator')
    else:
        discriminator_scheduler = StepLR(discriminator_optimizer, step_size=args.discrim_lr_step, gamma=args.discrim_gamma)
        print('step lr - discriminator')

    last_saved_epoch = 0 # default (if not loading from checkpoint)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_saved_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])

        print('loaded {}'.format(args.pretrained))

    if args.wandb_project:
        wandb.summary['model'] = str(model)
        wandb.watch(model, log='all', log_freq=args.log_interval*3)
    best_ssim = 0
    best_epoch = -1
    for epoch in range(last_saved_epoch + 1, args.epochs + 1):
        print('current learning rate: {:.2E}'.format(scheduler.get_last_lr()[0]))
        #with torch.autocast(device_type='cuda', dtype=torch.float16):
        train_loss, discriminator_accuracy = train(args=args, 
                        model=model, 
                        device=device, 
                        train_loader=train_loader, 
                        optimizer=optimizer, 
                        loss_fn=train_loss_fn, 
                        epoch=epoch, 
                        discriminator=discriminator, 
                        discriminator_optimizer=discriminator_optimizer)
        log_metrics = evaluate(model, val_loader, device, args)
        print(log_metrics)

        # save checkpoint
        torch.save(model.state_dict(), args.model_path.split('.')[0] + '_mostRecent.pt')

        if log_metrics[AVG_SSIM] > best_ssim: # we have a better SSIM than before
            best_ssim = log_metrics[AVG_SSIM]
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)
            print('save model')
            num_iterations_no_improvement = 0
        else:
            num_iterations_no_improvement += 1
        log_metrics.update({"train loss": train_loss, "learning rate": scheduler.get_last_lr()[0],\
                   "best ssim": best_ssim, "best epoch": best_epoch, "discriminator accuracy": discriminator_accuracy})
        if args.wandb_project:
            wandb.log(log_metrics) # send to wandb
        if num_iterations_no_improvement >= args.patience or epoch == args.epochs:
            print('best ssim = {} @ epoch {}'.format(best_ssim, best_epoch))
            print('we done')

            # log results - courtesy of ChatGPT
            now = datetime.datetime.now()
            # format the date as a string in the format 'YYYY-MM-DD_H:M'
            date_str = now.strftime('%Y-%m-%d_%H:%M')
            # create a folder with the date in its name
            folder_subname = f'{date_str}_results'
            folder_name = os.path.join(args.results_folder, folder_subname)
            os.makedirs(folder_name, exist_ok=True)
            log_filepath = os.path.join(folder_name, 'log.txt')
            # log data used
            train_data_info_filepath = os.path.join(folder_name, 'train_data.json')
            extra_data_info_filepath = os.path.join(folder_name, 'extraTrain_data.json')
            val_data_info_filepath = os.path.join(folder_name, 'val_data.json')
            for curr_dataset, curr_data_info_filepath in zip([all_datasets[0], all_datasets[1], val_dataset], 
                    [train_data_info_filepath, extra_data_info_filepath, val_data_info_filepath]):
                all_the_material_info = dict()
                for idx in range(len(curr_dataset)):
                    molecular_id = curr_dataset.get_mp_id(curr_dataset.filepaths[idx].split('/')[-1])
                    molecular_formula = curr_dataset.id_to_formulaStr[molecular_id]
                    curr_formula = curr_dataset.formulas[idx]
                    assert molecular_id not in all_the_material_info
                    all_the_material_info[molecular_id] = molecular_formula
                with open(curr_data_info_filepath, 'w') as fout:
                    json.dump(all_the_material_info, fout, indent=4)

            with open(log_filepath, 'w') as f:
                # write the args to the file
                for arg_name, arg_value in vars(args).items():
                    f.write(f'{arg_name}: {arg_value}\n')
                # write the PyTorch model architecture to the file
                f.write('\n{}\n\n'.format(str(model)))
                # write the loss and epoch it came at, and also the number of epochs we did
                f.write('best ssim = {} @ epoch {}\n'.format(best_ssim, best_epoch))
                f.write('stopped at epoch {}\n'.format(epoch))
            # write the model path
            model_path = os.path.join(folder_name, 'model.pt')
            torch.save(model.state_dict(), model_path)

            break

        # step LR schedules
        scheduler.step()
        discriminator_scheduler.step()

    return

if __name__ == "__main__":
    main()