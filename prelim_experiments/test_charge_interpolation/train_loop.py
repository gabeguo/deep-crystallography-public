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
from charge_density_dataset import ChargeDensityDataset
import math
import datetime

import wandb

import sys
sys.path.append('./models')
from deeper_charge_density_mlp import DeepChargeDensityRegressor
from dense_mlp import DenseChargeDensityRegressor

DEEP = 'deep'
DENSE = 'dense'

def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    print('Train Epoch: {}'.format(epoch))
    model.train()
    avg_train_loss = 0
    for batch_idx, (given_charges, pos, charge_at_pos) in enumerate(train_loader):
        given_charges = given_charges.to(device)
        pos = pos.to(device)
        charge_at_pos = charge_at_pos.to(device)

        optimizer.zero_grad()
        output = model(given_charges, pos)
        loss = loss_fn(output, charge_at_pos)
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item() / len(train_loader)
        if batch_idx % args.log_interval == 0:
            print('\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(given_charges), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            avg_final_linear_gradient = torch.mean(model.lastfc.weight.grad)
            std_final_linear_gradient = torch.std(model.lastfc.weight.grad)
            print('\t\tgradient at end: mean = {:.2E} std = {:.2E}'.format(avg_final_linear_gradient, std_final_linear_gradient))
            avg_final_linear_weight = torch.mean(model.lastfc.weight)
            std_final_linear_weight = torch.std(model.lastfc.weight)
            print('\t\tweight at end: mean = {:.2E} std = {:.2E}'.format(avg_final_linear_weight, std_final_linear_weight))
            avg_first_linear_gradient = torch.mean(model.initfc[0].weight.grad)
            std_first_linear_gradient = torch.std(model.initfc[0].weight.grad)
            print('\t\tgradient at start of regressor: mean = {:.2E}; std = {:.2E}'.format(\
                avg_first_linear_gradient, std_first_linear_gradient))
            avg_first_linear_weight = torch.mean(model.initfc[0].weight)
            std_first_linear_weight = torch.std(model.initfc[0].weight)
            print('\t\tweight at start of regressor: mean = {:.2E}; std = {:.2E}'.format(\
                avg_first_linear_weight, std_first_linear_weight))
            avg_first_linear_gradient_embedder = torch.mean(model.diffraction_embedder.initfc[0].weight.grad)
            std_first_linear_gradient_embedder = torch.std(model.diffraction_embedder.initfc[0].weight.grad)
            print('\t\tgradient at start of embedder: mean = {:.2E}; std = {:.2E}'.format(\
                avg_first_linear_gradient_embedder, std_first_linear_gradient_embedder))      
            avg_first_linear_weight_embedder = torch.mean(model.diffraction_embedder.initfc[0].weight)
            std_first_linear_weight_embedder = torch.std(model.diffraction_embedder.initfc[0].weight)
            print('\t\tweight at start of embedder: mean = {:.2E}; std = {:.2E}'.format(\
                avg_first_linear_weight_embedder, std_first_linear_weight_embedder))      
            avg_prediction = torch.mean(output)
            std_prediction = torch.std(output)
            print('\t\tprediction: mean = {:.4f} std = {:.4f}'.format(avg_prediction, std_prediction))
            avg_value = torch.mean(charge_at_pos)
            std_value = torch.std(charge_at_pos)
            print('\t\tground truth: mean = {:.4f} std = {:.4f}'.format(avg_value, std_value))
            
            if args.dry_run:
                break
        
        #print('linear layer:', model.lastfc[0].weight.grad)
        #print(output)
    print('\taverage train loss = {:.6f}'.format(avg_train_loss))
    return avg_train_loss


def test(model, device, test_loader, loss_fn, args, modality='test'):
    print('Test:')
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (given_charges, pos, charge_at_pos) in enumerate(test_loader):
            given_charges = given_charges.to(device)
            pos = pos.to(device)
            charge_at_pos = charge_at_pos.to(device)

            output = model(given_charges, pos)
            test_loss += loss_fn(output, charge_at_pos).item() / len(test_loader.dataset)  # sum up batch loss
            if idx % args.val_log_interval == 0:
                print('\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                idx * len(given_charges), len(test_loader.dataset),
                100. * idx / len(test_loader), test_loss * len(test_loader.dataset) / ((idx + 1) * args.test_batch_size)))
                # TODO: check output and compare to ground truth charge shape

    print('\n{} set: Average loss: {:.4f}\n'.format(modality, test_loss))

    return test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train the Charge Density Estimation Net')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
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
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of relative charge density bins (where charge density goes 0->1) to draw equally likely ' + \
                        'samples from (default: 10)')
    parser.add_argument('--uniform_charge_sample_prob', type=float, default=0.5,
                        help='Probability that we get a train sample with charge density uniformly sampled from [0, 1], ' + \
                        'rather than sampled from a random point on the cube')
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
    parser.add_argument('--model_path', type=str, default='/data/therealgabeguo/crystallography/models/charge_density_net.pt', metavar='MP',
                        help='path to save the current model at')
    parser.add_argument('--data_dir', type=str, default='/data/therealgabeguo/crystallography/charge_data_npy', metavar='D',
                        help='where is the data stored')
    parser.add_argument('--patience', type=int, default=5, metavar='P',
                        help='how long to train model before early stopping, if no improvement')
    parser.add_argument('--max_num_crystals', type=int, default=50,
                        help='how many train crystals to have in the dataset')
    parser.add_argument('--max_num_val_crystals', type=int, default=20,
                        help='how many val crystals to have in the dataset')
    parser.add_argument('--same_train_val', action='store_true',
                        help='whether to use the same training and val set (only for dev purposes)')   
    parser.add_argument('--model', type=str, default=DEEP, \
                        help='which model to use (options: {}, {}; default: {})'.format(DEEP, DENSE, DEEP))
    parser.add_argument('--num_blocks', type=int, default=4, \
                        help='how many two-layer blocks in regressor and embedder each (default: 4)')
    parser.add_argument('--num_freq', type=int, default=10, \
                        help='number of frequencies to use for Fourier features (default: 10)')
    parser.add_argument('--single_skip', action='store_true', \
                        help='whether to have skip connections that skip one block')
    parser.add_argument('--double_skip', action='store_true', \
                        help='whether to have skip connections that skip two blocks')
    parser.add_argument('--results_folder', type=str, default='/data/therealgabeguo/crystallography/run_logs', \
                        help='Where to write the results')
    parser.add_argument('--wandb_project', default='crystallography', \
                        help='database name for wandb')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

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

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    """
    Todo: create datasets
    """
    train_dataset = ChargeDensityDataset(os.path.join(args.data_dir, 'train'), \
                                         num_x_samples=args.num_x, num_y_samples=args.num_y, num_z_samples=args.num_z, \
                                         max_num_crystals=args.max_num_crystals, train=True, \
                                         n_bins=args.n_bins, uniform_charge_sample_prob=args.uniform_charge_sample_prob)
    val_dataset = ChargeDensityDataset(os.path.join(args.data_dir, 'train' if args.same_train_val else 'val'), \
                                         num_x_samples=args.num_x_val, num_y_samples=args.num_y_val, num_z_samples=args.num_z_val, \
                                         max_num_crystals=args.max_num_val_crystals, train=False)
    # test_dataset = ChargeDensityDataset(os.path.join(args.data_dir, 'test'), \
    #                                      num_x_samples=args.num_x, num_y_samples=args.num_y, num_z_samples=args.num_z)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    print('dataset loaded')

    if args.model == DENSE:
        model = DenseChargeDensityRegressor(num_blocks=args.num_blocks, num_freq=args.num_freq).to(device)
    else:
        model = DeepChargeDensityRegressor(\
            num_blocks=args.num_blocks, single_skip=args.single_skip, double_skip=args.double_skip\
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loss_fn = torch.nn.L1Loss(reduction='mean')
    test_loss_fn = torch.nn.L1Loss(reduction='sum')

    # create path for model
    if not os.path.exists(args.model_path):
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    num_iterations_no_improvement = 0
    # train loop
    if args.warm_restart:
        t_mult = max(1, int(args.gamma))
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                        T_0 = args.lr_step,# Number of iterations for the first restart
                                        T_mult = t_mult, # An integer factor to scale max learning rate by after a restart
                                        eta_min = args.min_lr) # Min learning rate
        print('cosine annealing')
    else:
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
        print('step lr')
    wandb.summary['model'] = str(model)
    wandb.watch(model, log='all', log_freq=args.log_interval*3)
    best_loss = math.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        print('current learning rate: {:.2E}'.format(scheduler.get_last_lr()[0]))
        train_loss = train(args, model, device, train_loader, optimizer, train_loss_fn, epoch)
        curr_loss = test(model, device, val_loader, test_loss_fn, args, modality='validation')
        torch.save(model.state_dict(), 'most_recent.pth')

        if curr_loss <= best_loss:
            best_loss = curr_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)
            print('save model')
            num_iterations_no_improvement = 0
        else:
            num_iterations_no_improvement += 1
        wandb.log({"train loss": train_loss, "val loss": curr_loss, "learning rate": scheduler.get_last_lr()[0],\
                   "best loss": best_loss, "best epoch": best_epoch})
        if num_iterations_no_improvement >= args.patience or epoch == args.epochs:
            print('best loss = {} @ epoch {}'.format(best_loss, best_epoch))
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

            with open(log_filepath, 'w') as f:
                # write the args to the file
                for arg_name, arg_value in vars(args).items():
                    f.write(f'{arg_name}: {arg_value}\n')
                # write the PyTorch model architecture to the file
                f.write('\n{}\n\n'.format(str(model)))
                # write the loss and epoch it came at, and also the number of epochs we did
                f.write('best loss = {} @ epoch {}\n'.format(best_loss, best_epoch))
                f.write('stopped at epoch {}\n'.format(epoch))
            # write the model path
            model_path = os.path.join(folder_name, 'model.pt')
            torch.save(model.state_dict(), model_path)

            break

        scheduler.step()

    return

if __name__ == "__main__":
    wandb.login()
    main()