# -*- coding: utf-8 -*-
"""network/utils.py

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

Utils file with plot function.
"""

import os
from matplotlib import pyplot as plt


def plot_predictions(inputs, predictions_input, path, update):
    """Plotting the cropped image and its concatenation with predictions to file path"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(1,2)
    # ax[1, 0].remove()
    # ax[1, 1].remove()

    for i in range(len(inputs)):
        ax[0].clear()
        ax[0].set_title('input')
        ax[0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0].set_axis_off()
        # ax[0, 1].clear()
        ax[1].set_title('input + prediction')
        ax[1].imshow(predictions_input[i], cmap=plt.cm.gray, interpolation='none')
        ax[1].set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1500)
    del fig


def plot_training(inputs, predictions_input, targets_displayed, predictions_displayed, path, update):
    """Plotting the cropped image, targets, predictions, predictions + input to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)
    # ax[1, 0].remove()
    # ax[1, 1].remove()

    for i in range(len(inputs)):
        ax[0, 0].clear()
        ax[0, 0].set_title('cropped image')
        ax[0, 0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('targets')
        ax[0, 1].imshow(targets_displayed[i], cmap=plt.cm.gray, interpolation='none')
        ax[0, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('predictions')
        ax[1, 0].imshow(predictions_displayed[i], cmap=plt.cm.gray, interpolation='none')
        ax[1, 0].set_axis_off()
        ax[1, 1].clear()
        ax[1, 1].set_title('predictions + input')
        ax[1, 1].imshow(predictions_input[i], cmap=plt.cm.gray, interpolation='none')
        ax[1, 1].set_axis_off()
        # fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1000)
    del fig
