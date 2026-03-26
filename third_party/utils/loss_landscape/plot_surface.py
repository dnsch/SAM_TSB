"""
Calculate and visualize the loss surface.
Usage example:
>>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""

import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluation
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import mpi4pytorch as mpi

sys.path.append(os.path.abspath(__file__ + "/../../"))
# TODO: clean up this file

# from sam.example.model.smooth_cross_entropy import smooth_crossentropy

import pdb


def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += "_[%s,%s,%d]" % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += "x[%s,%s,%d]" % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data:  # without data normalization
        surf_file += "_rawdata"
    if args.data_split > 1:
        surf_file += (
            "_datasplit=" + str(args.data_split) + "_splitidx=" + str(args.split_idx)
        )

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, "r")
        if (args.y and "ycoordinates" in f.keys()) or "xcoordinates" in f.keys():
            f.close()
            print("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, "a")
    f["dir_file"] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f["xcoordinates"] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f["ycoordinates"] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
    Calculate the loss values and accuracies of modified models in parallel
    using MPI reduce.
    """

    net.eval()

    f = h5py.File(surf_file, "r+" if rank == 0 else "r")
    if args.loss_name == "mse":
        losses, accuracies = [], []
    xcoordinates = f["xcoordinates"][:]
    ycoordinates = f["ycoordinates"][:] if "ycoordinates" in f.keys() else None

    if loss_key not in f.keys():
        shape = (
            xcoordinates.shape
            if ycoordinates is None
            else (len(xcoordinates), len(ycoordinates))
        )
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(
        losses, xcoordinates, ycoordinates, comm
    )

    print("Computing %d values for rank %d" % (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == "mse":
        criterion = nn.MSELoss()
    if args.loss_name == "L1":
        criterion = nn.L1Loss()
    # elif args.loss_name == "smooth_crossentropy":
    #     criterion = smooth_crossentropy

    # EDIT:
    # Get attention mode from args
    attention_mode = args.attention_mode

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # # Load the weights corresponding to those coordinates into the net
        # if args.dir_type == "weights":
        #     net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        # elif args.dir_type == "states":
        #     net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # EDIT:
        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == "weights":
            net_plotter.set_weights(
                net.module if args.ngpu > 1 else net,
                w,
                d,
                coord,
                attention_mode=attention_mode,  # Add this parameter
            )
        elif args.dir_type == "states":
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()

        # print(f"compute_keys weights: {net.compute_keys.weight}")
        # print(f"compute_queries weights: {net.compute_queries.weight}")
        # print(f"compute_values weights: {net.compute_values.weight}")

        if isinstance(criterion, nn.CrossEntropyLoss):
            loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda, model_name=args.model)
        elif isinstance(criterion, nn.MSELoss):
            loss = evaluation.eval_loss(net, criterion, dataloader, args.cuda, model_name=args.model)
        elif isinstance(criterion, nn.L1Loss):
            loss = evaluation.eval_loss(net, criterion, dataloader, args.cuda, model_name=args.model)

        print(f"loss: {loss}")
        for name, module in net.named_modules():
            if "dropout" in name.lower() or "norm" in name.lower():
                print(f"{name}: {module}")
        # elif callable(criterion) and criterion == smooth_crossentropy:
        #     loss, acc = evaluation.eval_loss(
        #         net, criterion, dataloader, args.cuda, args=args
        #     )
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        if isinstance(criterion, nn.CrossEntropyLoss):
            accuracies.ravel()[ind] = acc
        # elif callable(criterion) and criterion == smooth_crossentropy:
        #     accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses = mpi.reduce_max(comm, losses)
        if isinstance(criterion, nn.CrossEntropyLoss):
            accuracies = mpi.reduce_max(comm, accuracies)
        # elif callable(criterion) and criterion == smooth_crossentropy:
        #     accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            if isinstance(criterion, nn.CrossEntropyLoss):
                f[acc_key][:] = accuracies
            # elif callable(criterion) and criterion == smooth_crossentropy:
            #     f[acc_key][:] = accuracies
            f.flush()

        if isinstance(criterion, nn.CrossEntropyLoss):
            print(
                "Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f"
                % (
                    rank,
                    count,
                    len(inds),
                    100.0 * count / len(inds),
                    str(coord),
                    loss_key,
                    loss,
                    acc_key,
                    acc,
                    loss_compute_time,
                    syc_time,
                )
            )
        # elif callable(criterion) and criterion == smooth_crossentropy:
        #     print(
        #         "Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f"
        #         % (
        #             rank,
        #             count,
        #             len(inds),
        #             100.0 * count / len(inds),
        #             str(coord),
        #             loss_key,
        #             loss,
        #             acc_key,
        #             acc,
        #             loss_compute_time,
        #             syc_time,
        #         )
        #     )
        elif isinstance(criterion, nn.MSELoss):
            print(
                "Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f  \ttime=%.2f \tsync=%.2f"
                % (
                    rank,
                    count,
                    len(inds),
                    100.0 * count / len(inds),
                    str(coord),
                    loss_key,
                    loss,
                    loss_compute_time,
                    syc_time,
                )
            )
        elif isinstance(criterion, nn.L1Loss):
            print(
                "Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f  \ttime=%.2f \tsync=%.2f"
                % (
                    rank,
                    count,
                    len(inds),
                    100.0 * count / len(inds),
                    str(coord),
                    loss_key,
                    loss,
                    loss_compute_time,
                    syc_time,
                )
            )

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        if isinstance(criterion, nn.CrossEntropyLoss):
            accuracies = mpi.reduce_max(comm, accuracies)
        # elif callable(criterion) and criterion == smooth_crossentropy:
        #     accuracies = mpi.reduce_max(comm, accuracies)
        #
    total_time = time.time() - start_time
    print("Rank %d done!  Total time: %.2f Sync: %.2f" % (rank, total_time, total_sync))

    f.close()


###############################################################
#                          MAIN
###############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotting loss surface")

    # Loss landscape computation settings group

    ll_group = parser.add_argument_group(
        "Loss Landscape Computation",
        "Settings for loss landscape computation. For more details, visit: https://github.com/tomgoldstein/loss-landscape",
    )
    ll_group.add_argument("--mpi", "-m", action="store_true", help="use mpi")
    ll_group.add_argument("--cuda", "-c", action="store_true", help="use cuda")
    ll_group.add_argument(
        "--threads", default=2, type=int, metavar="N", help="number of threads"
    )
    ll_group.add_argument(
        "--ngpu",
        type=int,
        default=1,
        metavar="N",
        help="number of GPUs to use for each rank, useful for data parallel evaluation",
    )
    ll_group.add_argument(
        "--batch_size", default=256, type=int, metavar="N", help="minibatch size"
    )

    # data parameters

    data_group = parser.add_argument_group("Data Parameters", "Dataset configuration")
    data_group.add_argument(
        "--dataset",
        default="ts_dataset",
        metavar="NAME",
        help="cifar10 | imagenet | ts_dataset",
    )
    data_group.add_argument(
        "--datapath",
        default="cifar10/data",
        metavar="DIR",
        help="path to the dataset",
    )

    data_group.add_argument(
        "--ts_dataset_name",
        type=str,
        default="ETTh1",
        metavar="NAME",
        help="ETTh1/ETTh2/ETTm1/ETTm2/electricity/exchange_rate/national_illness/traffic/weather",
    )
    data_group.add_argument(
        "--raw_data",
        action="store_true",
        default=False,
        help="no data preprocessing",
    )
    data_group.add_argument(
        "--data_split",
        default=1,
        type=int,
        metavar="N",
        help="the number of splits for the dataloader",
    )
    data_group.add_argument(
        "--split_idx",
        default=0,
        type=int,
        metavar="N",
        help="the index of data splits for the dataloader",
    )
    data_group.add_argument(
        "--trainloader",
        default="",
        metavar="PATH",
        help="path to the dataloader with random labels",
    )
    data_group.add_argument(
        "--testloader",
        default="",
        metavar="PATH",
        help="path to the testloader with random labels",
    )

    # model parameters

    model_group = parser.add_argument_group("Model Parameters", "Model configuration")
    model_group.add_argument(
        "--model",
        default="samformer",
        metavar="NAME",
        help="model name",
    )
    model_group.add_argument(
        "--model_folder",
        default="",
        metavar="DIR",
        help="the common folder that contains model_file and model_file2",
    )
    model_group.add_argument(
        "--model_file",
        default="experiments/samformer/ETTh1/final_model_s2024.pt",
        metavar="PATH",
        help="path to the trained model file",
    )
    model_group.add_argument(
        "--model_file2",
        default="",
        metavar="PATH",
        help="use (model_file2 - model_file) as the xdirection",
    )
    model_group.add_argument(
        "--model_file3",
        default="",
        metavar="PATH",
        help="use (model_file3 - model_file) as the ydirection",
    )
    model_group.add_argument(
        "--loss_name",
        "-l",
        default="crossentropy",
        metavar="NAME",
        help="loss functions: crossentropy | mse",
    )



    # direction parameters

    dir_group = parser.add_argument_group(
        "Direction Parameters",
        "Parameters for defining loss landscape directions",
    )
    dir_group.add_argument(
        "--dir_file",
        default="",
        metavar="PATH",
        help="specify the name of direction file, or the path to an existing direction file",
    )
    dir_group.add_argument(
        "--dir_type",
        default="weights",
        metavar="TYPE",
        help="direction type: weights | states (including BN's running_mean/var)",
    )
    dir_group.add_argument(
        "--x",
        default="-1:1:51",
        metavar="MIN:MAX:NUM",
        help="A string with format xmin:x_max:xnum",
    )
    dir_group.add_argument(
        "--y",
        default=None,
        metavar="MIN:MAX:NUM",
        help="A string with format ymin:ymax:ynum",
    )
    dir_group.add_argument(
        "--xnorm",
        default="",
        metavar="TYPE",
        help="direction normalization: filter | layer | weight",
    )
    dir_group.add_argument(
        "--ynorm",
        default="",
        metavar="TYPE",
        help="direction normalization: filter | layer | weight",
    )
    dir_group.add_argument(
        "--xignore",
        default="",
        metavar="TYPE",
        help="ignore bias and BN parameters: biasbn",
    )
    dir_group.add_argument(
        "--yignore",
        default="",
        metavar="TYPE",
        help="ignore bias and BN parameters: biasbn",
    )
    dir_group.add_argument(
        "--same_dir",
        action="store_true",
        default=False,
        help="use the same random direction for both x-axis and y-axis",
    )
    dir_group.add_argument(
        "--idx",
        default=0,
        type=int,
        metavar="N",
        help="the index for the repeatness experiment",
    )
    dir_group.add_argument(
        "--surf_file",
        default="",
        metavar="PATH",
        help="customize the name of surface file, could be an existing file.",
    )
    dir_group.add_argument(
        "--hessian_directions",
        action="store_true",
        default=False,
        help="create hessian eigenvectors directions h5 file",
    )
    # EDIT:
    dir_group.add_argument(
        "--attention_mode",
        default="all",
        metavar="MODE",
        choices=["all", "attention_only", "no_attention"],
        help="Which weights to vary: all | attention_only | no_attention",
    )

    # plot parameters

    plot_group = parser.add_argument_group(
        "Plot Parameters", "Visualization and plotting options"
    )
    plot_group.add_argument(
        "--proj_file",
        default="",
        metavar="PATH",
        help="the .h5 file contains projected optimization trajectory.",
    )
    plot_group.add_argument(
        "--loss_max",
        default=5,
        type=float,
        metavar="VAL",
        help="Maximum value to show in 1D plot",
    )
    plot_group.add_argument(
        "--vmax", default=10, type=float, metavar="VAL", help="Maximum value to map"
    )
    plot_group.add_argument(
        "--vmin", default=0.1, type=float, metavar="VAL", help="Minimum value to map"
    )
    plot_group.add_argument(
        "--vlevel",
        default=0.5,
        type=float,
        metavar="VAL",
        help="plot contours every vlevel",
    )
    plot_group.add_argument(
        "--show", action="store_true", default=False, help="show plotted figures"
    )
    plot_group.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="use log scale for loss values",
    )
    plot_group.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="plot figures after computation",
    )

    # TODO: not sure if these are needed
    #
    # # Model architecture parameters
    #
    # arch_group = parser.add_argument_group(
    #     "Architecture Parameters", "Model architecture configuration"
    # )
    # arch_group.add_argument(
    #     "--depth", default=16, type=int, metavar="N", help="Number of layers."
    # )
    # arch_group.add_argument(
    #     "--width_factor",
    #     default=8,
    #     type=int,
    #     metavar="N",
    #     help="How many times wider compared to normal ResNet.",
    # )
    # arch_group.add_argument(
    #     "--dropout", default=0.0, type=float, metavar="RATE", help="Dropout rate."
    # )
    # arch_group.add_argument(
    #     "--label_smoothing",
    #     default=0.1,
    #     type=float,
    #     metavar="VAL",
    #     help="Use 0.0 for no label smoothing.",
    # )

    # samformer parameters

    samformer_group = parser.add_argument_group(
        "SAMFormer Parameters", "Parameters specific to SAMFormer model"
    )
    samformer_group.add_argument(
        "--seq_len",
        default=512,
        type=int,
        metavar="N",
        help="denotes input history length",
    )
    samformer_group.add_argument(
        "--horizon",
        default=96,
        type=int,
        metavar="N",
        help="denotes prediction length",
    )
    samformer_group.add_argument("--use_revin", type=bool, default=True, metavar="BOOL")
    samformer_group.add_argument("--hid_dim", type=int, default=16, metavar="N")
    samformer_group.add_argument("--input_dim", type=int, default=3, metavar="N")
    samformer_group.add_argument("--output_dim", type=int, default=1, metavar="N")

 # patchtst parameters
    patchtst_group = parser.add_argument_group(
        "PatchTST Parameters", "Parameters specific to PatchTST model"
    )
    patchtst_group.add_argument(
        "--enc_in", type=int, default=7, metavar="N",
        help="encoder input size (number of channels)"
    )
    patchtst_group.add_argument(
        "--e_layers", type=int, default=3, metavar="N",
        help="number of encoder layers"
    )
    patchtst_group.add_argument(
        "--n_heads", type=int, default=4, metavar="N",
        help="number of attention heads"
    )
    patchtst_group.add_argument(
        "--d_model", type=int, default=16, metavar="N",
        help="dimension of model"
    )
    patchtst_group.add_argument(
        "--d_ff", type=int, default=128, metavar="N",
        help="dimension of feedforward network"
    )
    patchtst_group.add_argument(
        "--dropout", type=float, default=0.2, metavar="RATE",
        help="dropout rate"
    )
    patchtst_group.add_argument(
        "--fc_dropout", type=float, default=0.2, metavar="RATE",
        help="fully connected dropout rate"
    )
    patchtst_group.add_argument(
        "--head_dropout", type=float, default=0.0, metavar="RATE",
        help="head dropout rate"
    )
    patchtst_group.add_argument(
        "--patch_len", type=int, default=16, metavar="N",
        help="patch length"
    )
    patchtst_group.add_argument(
        "--stride", type=int, default=8, metavar="N",
        help="stride for patching"
    )
    patchtst_group.add_argument(
        "--padding_patch", type=str, default="end",
        help="padding patch: None or 'end'"
    )
    patchtst_group.add_argument(
        "--decomposition", action="store_true", default=False,
        help="use series decomposition"
    )
    patchtst_group.add_argument(
        "--kernel_size", type=int, default=25, metavar="N",
        help="kernel size for decomposition"
    )
    patchtst_group.add_argument(
        "--individual", action="store_true", default=False,
        help="use individual head for each channel"
    )
    patchtst_group.add_argument(
        "--attn_dropout", type=float, default=0.0, metavar="RATE",
        help="attention dropout rate"
    )
    patchtst_group.add_argument(
        "--res_attention", action="store_true", default=True,
        help="use residual attention"
    )
    patchtst_group.add_argument(
        "--pre_norm", action="store_true", default=False,
        help="use pre-normalization"
    )
    patchtst_group.add_argument(
        "--pe", type=str, default="zeros",
        help="positional encoding type: zeros | normal | uniform | lin1d | exp1d | lin2d | exp2d | pos2d"
    )
    patchtst_group.add_argument(
        "--learn_pe", action="store_true", default=True,
        help="learn positional encoding"
    )
    patchtst_group.add_argument(
        "--head_type", type=str, default="flatten",
        help="head type: flatten | individual"
    )

    args = parser.parse_args()

    args = parser.parse_args()

    torch.manual_seed(123)
    # --------------------------------------------------------------------------
    # Environment setup
    # --------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception(
                "User selected cuda option, but cuda is not available on this machine"
            )
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print(
            "Rank %d use GPU %d of %d GPUs on %s"
            % (rank, torch.cuda.current_device(), gpu_count, socket.gethostname())
        )

    # --------------------------------------------------------------------------
    # Check plotting resolution
    # --------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(":")]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(":")]
            assert args.ymin and args.ymax and args.ynum, (
                "You specified some arguments for the y axis, but not all"
            )
    except:
        raise Exception(
            "Improper format for x- or y-coordinates. Try something like -1:1:51"
        )

    # --------------------------------------------------------------------------
    # Load models and extract parameters
    # --------------------------------------------------------------------------

    if args.model == "samformer":
        node_num = None
        # num_channels = dataloader["train_loader"].dataset[0][0].shape[0]
        num_channels = 7

        net = model_loader.samformer_load(
            node_num,
            args.input_dim,
            args.output_dim,
            num_channels,
            args.seq_len,
            args.hid_dim,
            args.horizon,
            args.use_revin,
            args.model,
            args.model_file,
        )
    elif args.model == "patchtst":
        # Handle padding_patch: convert string "None" to actual None
        padding_patch = args.padding_patch if args.padding_patch != "None" else None
        
        net = model_loader.patchtst_load(
            enc_in=args.enc_in,
            seq_len=args.seq_len,
            pred_len=args.horizon,
            e_layers=args.e_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,
            patch_len=args.patch_len,
            stride=args.stride,
            padding_patch=padding_patch,
            decomposition=args.decomposition,
            kernel_size=args.kernel_size,
            individual=args.individual,
            model_file=args.model_file,
            attn_dropout=args.attn_dropout,
            res_attention=args.res_attention,
            pre_norm=args.pre_norm,
            pe=args.pe,
            learn_pe=args.learn_pe,
            head_type=args.head_type,
        )



    elif args.model == "wideresnet":
        depth = args.depth
        width_factor = args.width_factor
        dropout = args.dropout
        in_channels = 3
        labels = 10
        #
        # net = model_loader.wideresnet_load(
        #     depth, width_factor, dropout, in_channels, labels
        # )
        net = model_loader.wideresnet_load(
            args.model,
            args.model_file,
            depth,
            width_factor,
            dropout,
            in_channels,
            labels,
        )
    else:
        net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net)  # initial parameters
    s = copy.deepcopy(net.state_dict())  # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    # --------------------------------------------------------------------------
    # Setup the direction file and the surface file
    # --------------------------------------------------------------------------

    dir_file = net_plotter.name_direction_file(args)  # name the direction file
    if not (args.hessian_directions):
        if rank == 0:
            net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    if args.hessian_directions:
        # TODO: check this hard coded path
        d = net_plotter.load_directions(
            "cifar10/trained_nets/samfo/final_model_s2024.pt_weights_xignore=biasbn_xnorm=dweights_yignore=biasbn_ynorm=dweights_hessian_directions.h5"
        )
    else:
        d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(
            proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1])
        )
        print("cosine similarity between x-axis and y-axis: %f" % similarity)

    # --------------------------------------------------------------------------
    # Setup dataloader
    # --------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    if rank == 0 and args.dataset == "cifar10":
        torchvision.datasets.CIFAR10(
            root=args.dataset + "/data", train=True, download=True
        )

    mpi.barrier(comm)

    trainloader, testloader = dataloader.load_dataset(
        args,
        args.dataset,
        args.datapath,
        args.batch_size,
        args.threads,
        args.raw_data,
        args.data_split,
        args.split_idx,
        args.trainloader,
        args.testloader,
    )

    # --------------------------------------------------------------------------
    # Start the computation
    # --------------------------------------------------------------------------
    crunch(
        surf_file,
        net,
        w,
        s,
        d,
        trainloader,
        "train_loss",
        "train_acc",
        comm,
        rank,
        args,
    )

    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    # --------------------------------------------------------------------------
    # Plot figures
    # --------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(
                surf_file, dir_file, args.proj_file, "train_loss", args.show
            )
        elif args.y:
            plot_2D.plot_2d_contour(
                surf_file, "train_loss", args.vmin, args.vmax, args.vlevel, args.show
            )
        else:
            plot_1D.plot_1d_loss_err(
                surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show
            )
