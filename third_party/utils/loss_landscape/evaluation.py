"""
The calculation to be performed at each point (modified model), evaluating
the loss value, accuracy and eigen values of the hessian matrix
"""

# TODO: clean up this file

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import random

# plotting for debugging model with changed weights performance
# from src.utils.functions import (
#     plot_stats,
#     branch_plot,
#     plot_mean_per_day,
#     mean_branch_plot,
# )
import os, sys

sys.path.append(os.path.abspath(__file__ + "/../../"))

# from sam.example.model.smooth_cross_entropy import smooth_crossentropy


def eval_loss(net, criterion, loader, use_cuda=True, args=None, model_name=None):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
        model_name: name of the model (for handling different input formats)
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    losses = []
    total = 0

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for inputs, targets in loader:
                batch_size = inputs.size(0)
                total += batch_size
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for inputs, targets in loader:
                batch_size = inputs.size(0)
                total += batch_size
                inputs, targets = inputs.to(device), targets.to(device)

                # PatchTST expects [Batch, Seq_len, Channels]
                # Dataloader outputs [Batch, Channels, Seq_len]
                if model_name == "patchtst":
                    inputs = inputs.permute(0, 2, 1)

                outputs = net(inputs)

                # Transpose outputs back for loss calculation if needed
                if model_name == "patchtst":
                    outputs = outputs.permute(0, 2, 1)

                loss = criterion(outputs, targets)
                losses.append(loss.item())
                total_loss += loss.item() * batch_size

        elif isinstance(criterion, nn.L1Loss):
            for inputs, targets in loader:
                batch_size = inputs.size(0)
                total += batch_size
                inputs, targets = inputs.to(device), targets.to(device)

                if model_name == "patchtst":
                    inputs = inputs.permute(0, 2, 1)

                outputs = net(inputs)

                if model_name == "patchtst":
                    outputs = outputs.permute(0, 2, 1)

                loss = criterion(outputs, targets)
                losses.append(loss.item())
                total_loss += loss.item() * batch_size

    if isinstance(criterion, nn.CrossEntropyLoss):
        return total_loss / total, 100.0 * correct / total
    elif isinstance(criterion, nn.MSELoss):
        return total_loss / total
    elif isinstance(criterion, nn.L1Loss):
        return total_loss / total
