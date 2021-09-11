# -*- coding: utf-8 -*-
"""network/train.py

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

Training workflow.
"""

import os
import numpy as np
import torch
import torch.utils.data
from utils import plot_training
from architectures import Ex5Cnn
from torch.utils.tensorboard import SummaryWriter
import tqdm


def process_outputs_training(inputs, outputs, targets):
    """
    :param inputs:
    :param outputs:
    :return: predictions as 1D numpy array and cropped image where the unknown pixels are replaced with predictions.
    """
    predictions = []
    inputs_plus_predictions = []
    targets_displayed = []
    predictions_displayed = []

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

        # concatenate targets and known array
        known_array = inputs[sample, 1, :, :].detach().clone()
        known_array[mask == 0] = targets[sample][:len(prediction)]
        targets_displayed.append(known_array.cpu().detach().numpy().astype(np.uint8))

        # concatenate predictions and known array
        prediction_known_array = inputs[sample, 1, :, :].detach().clone()
        prediction_known_array[mask == 0] = prediction
        predictions_displayed.append(prediction_known_array.cpu().detach().numpy().astype(np.uint8))

    predictions_max_shape = max([len(prediction) for prediction in predictions])

    stacked_predictions = torch.zeros(size=(inputs.shape[0], predictions_max_shape), dtype=torch.float32)
    for idx, prediction in enumerate(predictions):
        stacked_predictions[idx, :len(prediction)] = prediction

    return stacked_predictions, inputs_plus_predictions, targets_displayed, predictions_displayed


def get_predictions_for_evaluation(inputs, outputs, targets):
    predictions = torch.zeros(size=(inputs.shape[0], targets.shape[1]), dtype=torch.float32)
    for sample in range(inputs.shape[0]):
        image = outputs[sample, 0, :, :]
        mask = inputs[sample, 1, :, :]
        prediction = image[mask == 0]
        predictions[sample, :len(prediction)] = prediction
    return predictions


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            inputs, means, stds, sample_ids, targets = data
            # inputs = inputs.to(device)
            # targets = targets.to(device)

            # Get outputs for network
            outputs = model(inputs)
            # de-normalize
            for sample in range(outputs.shape[0]):
                outputs[sample, 0, :, :] *= stds[sample]
                outputs[sample, 0, :, :] += means[sample]

            # outputs = get_predictions_for_evaluation(inputs, outputs, targets)
            predictions_tensor, inputs_plus_predictions, targets_displayed, predictions_displayed = process_outputs_training(
                inputs, outputs,
                targets)
            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(output, target) for output, target in zip(predictions_tensor, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def train_net(results_path, hyperparameters, trainloader, valloader, testloader, plotpath):
    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    # Create Network
    net = Ex5Cnn(**hyperparameters['network_config'])

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparameters['learningrate'],
                                 weight_decay=hyperparameters['weight_decay'])

    print_stats_at = 1e2  # print status to tensorboard every x updates
    plot_at = 1e4  # plot every x updates
    validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    device = hyperparameters['device']
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm.tqdm(total=hyperparameters['n_updates'], desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    n_updates = hyperparameters['n_updates']
    while update < n_updates:
        for data in trainloader:
            inputs, means, stds, sample_ids, targets = data
            # inputs = inputs.to(device)
            # targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = net(inputs)
            # de-normalize outputs (note that known_array ([sample, 1, :, :] was not normalized)
            for sample in range(outputs.shape[0]):
                outputs[sample, 0, :, :] *= stds[sample]
                outputs[sample, 0, :, :] += means[sample]

            predictions_tensor, inputs_plus_predictions, targets_displayed, predictions_displayed = process_outputs_training(
                inputs, outputs,
                targets)
            # Calculate loss, do backward pass, and update weights
            loss = mse(predictions_tensor, targets)
            loss.backward()
            optimizer.step()

            # de-normalize inputs (should be done after the backward() bcs. of the gradients)
            for sample in range(inputs.shape[0]):
                inputs[sample, 0, :, :] *= stds[sample]
                inputs[sample, 0, :, :] += means[sample]

            predictions_tensor, inputs_plus_predictions, targets_displayed, predictions_displayed = process_outputs_training(
                inputs, outputs,
                targets)

            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="network/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
            # plot results
            if update % plot_at == 0:
                plot_training(inputs=inputs.detach().cpu().numpy(),
                              predictions_input=inputs_plus_predictions, targets_displayed=targets_displayed,
                              predictions_displayed=predictions_displayed,
                              path=plotpath, update=update)
                # validation
                if update % validate_at == 0 and update > 0:
                    val_loss = evaluate_model(net, dataloader=valloader, device=device)
                    writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                    # Add weights as arrays to tensorboard
                    for i, param in enumerate(net.parameters()):
                        writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                             global_step=update)
                    # Add gradients as arrays to tensorboard
                    for i, param in enumerate(net.parameters()):
                        writer.add_histogram(tag=f'validation/gradients_{i}',
                                             values=param.grad.cpu(),
                                             global_step=update)
                        # Save best model for early stopping
                    if best_validation_loss > val_loss:
                        best_validation_loss = val_loss
                        torch.save(net, os.path.join(results_path, 'best_model.pt'))

        update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        update_progress_bar.update()
        # Increment update counter, exit if maximum number of updates is reached
        update += 1
        if update >= n_updates:
            break

    update_progress_bar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"network loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'scores.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
