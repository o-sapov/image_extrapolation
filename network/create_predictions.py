# -*- coding: utf-8 -*-
"""network/create_predictions.py

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
Functions for creating predictions.
"""


import torch
from utils import plot_predictions
import os
import dill as pkl
import numpy as np


def process_outputs_predictions(inputs, outputs):
    """
    :param inputs:
    :param outputs:
    :return: predictions as 1D numpy array (no trailing zeros, different to process_outputs function in train.py) and cropped image where the unknown pixels are replaced with predictions.
    """
    predictions = []
    inputs_plus_predictions = []
    for sample in range(inputs.shape[0]):
        # get predictions
        image = outputs[sample, 0, :, :]
        mask = inputs[sample, 1, :, :]
        prediction = image[mask == 0]
        predictions.append(prediction)

        # concatenate inputs and predictions
        cropped_image = inputs[sample, 0, :, :].detach().clone()
        cropped_image[mask == 0] = prediction
        inputs_plus_predictions.append(cropped_image.cpu().detach().numpy().astype(np.uint8))

    return predictions, inputs_plus_predictions


def create_predictions(dataset, results_path, plotpath):
    # Load Network
    model = torch.load(os.path.join('..', 'model.pt'))

    plot_at = 1e3  # plot every x updates
    predictions_list = []
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            inputs, means, stds, sample_ids = data
            # Get outputs for network
            outputs = model(inputs)
            # de-normalize
            for sample in range(outputs.shape[0]):
                outputs[sample, 0, :, :] *= stds[sample]
                outputs[sample, 0, :, :] += means[sample]

                inputs[sample, 0, :, :] *= stds[sample]
                inputs[sample, 0, :, :] += means[sample]

            # get predictions and its concatenation with inputs
            predictions, predictions_input = process_outputs_predictions(inputs, outputs)
            # convert tensor to numpy array with datatyp uint8
            predictions = [prediction.cpu().detach().numpy().astype(np.uint8) for prediction in predictions]
            # add to the list in order not to loos current predictions
            for prediction in predictions:
                predictions_list.append(prediction)

            if idx % plot_at == 0:
                plot_predictions(inputs=inputs.detach().cpu().numpy(),
                                 predictions_input=predictions_input, path=plotpath, update=idx)

    with open(os.path.join(results_path, 'predict', 'predictions.pkl'), "wb") as ufh:
        pkl.dump(predictions_list, ufh)

        print('Finished creating predictions!')
