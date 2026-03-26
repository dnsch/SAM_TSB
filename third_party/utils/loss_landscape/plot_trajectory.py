"""
Plot the optimization path in the space spanned by principle directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import model_loader
import net_plotter
from projection import setup_PCA_directions, project_trajectory
import plot_2D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot optimization trajectory")
    parser.add_argument("--dataset", default="cifar10", help="dataset")
    parser.add_argument("--model", default="resnet56", help="trained models")
    parser.add_argument(
        "--model_folder", default="", help="folders for models to be projected"
    )
    parser.add_argument(
        "--dir_type",
        default="weights",
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""",
    )
    parser.add_argument(
        "--ignore", default="", help="ignore bias and BN paras: biasbn (no bias or bn)"
    )
    parser.add_argument(
        "--prefix", default="model_", help="prefix for the checkpint model"
    )
    parser.add_argument(
        "--suffix", default=".t7", help="prefix for the checkpint model"
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="min index of epochs"
    )
    parser.add_argument(
        "--max_epoch", default=300, type=int, help="max number of epochs"
    )
    parser.add_argument(
        "--save_epoch", default=1, type=int, help="save models every few epochs"
    )
    parser.add_argument(
        "--dir_file", default="", help="load the direction file for projection"
    )
    parser.add_argument(
        "--surf_file", default="", help="load the direction file for projection"
    )

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # load the final model
    # --------------------------------------------------------------------------
    last_model_file = (
        args.model_folder + "/" + args.prefix + str(args.max_epoch) + args.suffix
    )
    if args.model == "samformer":
        node_num = None
        input_dim = 3
        output_dim = 1
        # num_channels = dataloader["train_loader"].dataset[0][0].shape[0]
        num_channels = 7
        seq_len = 512
        hid_dim = 16
        horizon = 96
        use_revin = True

        net = model_loader.samformer_load(
            node_num,
            input_dim,
            output_dim,
            num_channels,
            seq_len,
            hid_dim,
            horizon,
            use_revin,
            args.dataset,
            args.model,
            last_model_file,
        )
    else:
        net = model_loader.load(args.dataset, args.model, last_model_file)

    w = net_plotter.get_weights(net)
    s = net.state_dict()

    # --------------------------------------------------------------------------
    # collect models to be projected
    # --------------------------------------------------------------------------
    model_files = []
    for epoch in range(
        args.start_epoch, args.max_epoch + args.save_epoch, args.save_epoch
    ):
        model_file = args.model_folder + "/" + args.prefix + str(epoch) + args.suffix
        assert os.path.exists(model_file), "model %s does not exist" % model_file
        model_files.append(model_file)

    # --------------------------------------------------------------------------
    # load or create projection directions
    # --------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        dir_file = setup_PCA_directions(args, model_files, w, s)

    # --------------------------------------------------------------------------
    # projection trajectory to given directions
    # --------------------------------------------------------------------------
    proj_file = project_trajectory(
        args,
        dir_file,
        w,
        s,
        args.dataset,
        args.model,
        model_files,
        args.dir_type,
        "cos",
    )
    # plot_2D.plot_trajectory(proj_file, dir_file)
    plot_2D.plot_contour_trajectory(
        args.surf_file, dir_file, proj_file, surf_name="train_loss"
    )
