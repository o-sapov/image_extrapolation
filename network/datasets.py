# -*- coding: utf-8 -*-
"""network/datasets.py

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

Datasets file.
"""

import numpy as np
from torch.utils.data import Dataset
import torch
import dill as pkl
import os


class Ex5Data(Dataset):
    def __init__(self, train):
        # load dataset
        dataset = os.path.join('..', 'dataset', 'out', 'dataset.pkl')
        with open(dataset, 'rb') as f:
            dataset = pkl.load(f)

        self.target_arrays = None
        if train:
            # load targets
            targets = os.path.join('..', 'dataset', 'out', 'targets.pkl')
            with open(targets, 'rb') as f:
                targets = pkl.load(f)

            self.target_arrays = tuple(targets)

        self.input_arrays = dataset["input_arrays"]
        self.known_arrays = dataset["known_arrays"]

    def __getitem__(self, index):
        inputs = self.input_arrays[index]
        inputs = np.array(inputs, dtype=np.float32)
        mean = inputs.mean()
        std = inputs.std()
        inputs[:] -= mean
        inputs[:] /= std
        if self.target_arrays == None:
            return inputs, self.known_arrays[index], mean, std, index
        else:
            return inputs, self.known_arrays[index], mean, std, index, self.target_arrays[index]

    def __len__(self):
        return len(self.input_arrays)


def ex5_collate_fn(batch_as_list: list):
    n_samples = len(batch_as_list)
    n_feature_channels = 2

    max_X = max([sample[0].shape[0] for sample in batch_as_list])
    max_Y = max([sample[1].shape[0] for sample in batch_as_list])
    # stack images of input array to the dimensions (n_samples, n_feature_channels, max_X, max_Y)

    inputs = torch.zeros(size=(n_samples, n_feature_channels, max_X, max_Y), dtype=torch.float32)
    # stack input_array
    for sample in range(n_samples):
        image = batch_as_list[sample][0]
        inputs[sample, 0, :, :] = torch.tensor(image)

    # add known_array to the previous tensor as second channel
    for sample in range(n_samples):
        image = batch_as_list[sample][1]
        inputs[sample, 1, :, :] = torch.tensor(image)

    means = [batch_as_list[sample][2] for sample in range(n_samples)]
    stds = [batch_as_list[sample][3] for sample in range(n_samples)]
    indices = [batch_as_list[sample][4] for sample in range(n_samples)]

    # If no targets are here, this is the "create_predictions" mode
    if len(batch_as_list[0]) == 5:
        return inputs, means, stds, indices
    # Otherwise "train" mode
    elif len(batch_as_list[0]) == 6:
        # Stack target_arrays. As they are provided as 1D we need a different approach then above
        target_arrays = [sample[5] for sample in batch_as_list]
        max_array_len = np.max([len(array) for array in target_arrays])
        # Allocate a tensor that can fit all padded sequences
        stacked_target_arrays = torch.zeros(size=(n_samples, max_array_len), dtype=torch.float32)
        # Write the sequences into the tensor stacked_target_arrays. The values which extend the length of the array will be padded with 0
        for i, array in enumerate(target_arrays):
            stacked_target_arrays[i, :len(array)] = torch.from_numpy(array)

        return inputs, means, stds, indices, stacked_target_arrays
    else:
        print("The length of the batch_as_list is undefined!")
