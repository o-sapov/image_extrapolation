# -*- coding: utf-8 -*-
"""network/main.py

Author -- Oleksii Sapov
Contact -- sapov@gmx.at
Date -- 25.07.2021


###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################
Main file for image extrapolation.
"""

import os
import numpy as np
import torch
import torch.utils.data
from datasets import Ex5Data
from datasets import ex5_collate_fn
import dill as pkl
import shutil
from create_predictions import create_predictions
from train import train_net

NUM_WORKERS = 0


def clean_up_dir(storage_path):
    shutil.rmtree(storage_path, ignore_errors=True)


def create_dataloaders(results_path: str, train: bool):
    # load ex5 dataset
    dataset = Ex5Data(train)
    # Split dataset into network, validation, and test set
    dataset_length = len(dataset)
    shuffled_indices = np.random.permutation(dataset_length)
    trainingset_inds = np.arange(int(dataset_length * (3 / 5)))
    testset_inds = np.arange(int(dataset_length * (3 / 5)),
                             int(dataset_length * (4 / 5)))
    validationset_inds = np.arange(int(dataset_length * (4 / 5)),
                                   dataset_length)
    # save indices
    indices = dict(trainingset=trainingset_inds, testset=testset_inds, validationset=validationset_inds)
    with open(os.path.join(results_path, 'indices.pkl'), "wb") as ufh:
        pkl.dump(indices, ufh)

    trainingset = torch.utils.data.Subset(dataset, indices=trainingset_inds)
    validationset = torch.utils.data.Subset(dataset, indices=testset_inds)
    testset = torch.utils.data.Subset(dataset, indices=validationset_inds)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=20, shuffle=True, num_workers=NUM_WORKERS,
                                              collate_fn=ex5_collate_fn)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
                                            collate_fn=ex5_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
                                             collate_fn=ex5_collate_fn)
    return trainloader, valloader, testloader


def main(results_path, train: int, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cpu")):
    # convert 1 or 0 to boolean
    train = bool(train)

    hyperparameters = {'network_config': network_config, 'learningrate': learningrate, 'weight_decay': weight_decay,
                       'n_updates': n_updates, 'device': device}

    if train:
        # Prepare a path to plot to
        plotpath = os.path.join(results_path, 'train', 'plots')
        clean_up_dir(plotpath)
        os.makedirs(plotpath, exist_ok=True)

        # create dataloaders for training, validation and testing
        trainloader, valloader, testloader = create_dataloaders(results_path=results_path, train=train)

        train_net(results_path=os.path.join(results_path, 'train'), hyperparameters=hyperparameters,
                  trainloader=trainloader,
                  valloader=valloader, testloader=testloader, plotpath=plotpath)
    else:
        # Prepare a path to plot to
        plotpath = os.path.join(results_path, 'predict', 'plots')
        clean_up_dir(plotpath)
        os.makedirs(plotpath, exist_ok=True)
        dataset = Ex5Data(train)
        # Create dataloader of the whole dataset
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=NUM_WORKERS,
                                                     collate_fn=ex5_collate_fn)

        create_predictions(dataset=dataset_loader, results_path=results_path, plotpath=plotpath)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
