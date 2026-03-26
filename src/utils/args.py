import argparse

"""
File to build model and configuration specific arguments
"""

# TODO: check if we can change those BOOL types, see
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

# TODO: add vgg arg group
# then add this to the _add_vgg_args
#
#
# Override after all groups are added
# model_group = _get_argument_group(parser, "Model")
# if model_group:
#     # Find and modify the existing argument
#     for action in model_group._group_actions:
#         if action.dest == "loss_name":
#             action.default = "cross-entropy"
#             action.help = "loss functions: cross-entropy"
#             break


# Helper functions
# Add argument to an existing group helper
def _get_argument_group(parser, title):
    """Retrieve an existing argument group by title."""
    for group in parser._action_groups:
        if group.title == title:
            return group
    return None


# taken from Maxim's answer on
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse


# TODO: change all bool (store_true etc.) to this
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _override_argument_default(parser, dest, new_default, group_title=None):
    """Override the default value of an existing argument.

    Args:
        parser: The argument parser
        dest: The destination name of the argument to modify
        new_default: The new default value
        group_title: Optional group title to search in (searches all if None)
    """
    groups = parser._action_groups
    if group_title:
        groups = [g for g in groups if g.title == group_title]

    for group in groups:
        for action in group._group_actions:
            if action.dest == dest:
                action.default = new_default
                return True
    return False


# Base configuration (used in all experiments)


def get_base_config():
    """Base configuration parser with core training arguments."""
    parser = argparse.ArgumentParser(
        description="Configurations for Model Training and Evaluation:",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_group = parser.add_argument_group("Model", "Model settings")

    model_group.add_argument(
        "--mode",
        type=str,
        default="train",
        metavar="MODE",
        help="operation mode: train | test | eval",
    )
    model_group.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="N",
        help="random seed for reproducibility",
    )
    # TODO: add additional loss functions
    model_group.add_argument(
        "--loss_name",
        "-l",
        default="mse",
        metavar="LOSS",
        help="loss functions: mse | mae | rmse | mape",
    )

    model_group.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["rmse", "mae", "mape"],
        choices=["mse", "rmse", "mae", "mape"],
        metavar="METRIC",
        help="Additional metrics to compute (can specify multiple): mse | mae | rmse | mape",
    )

    # Dataset
    dataset_group = parser.add_argument_group("Dataset", "Dataset settings")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="",
        required=True,
        metavar="NAME",
        help="name of the dataset to use",
    )
    dataset_group.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        metavar="RATIO",
        help="ratio of data to use for training (0.0-1.0)",
    )
    dataset_group.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        metavar="RATIO",
        help="ratio of data to use for validation (0.0-1.0)",
    )
    dataset_group.add_argument(
        "--raw_data",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="do not normalize data",
    )

    dataset_group.add_argument(
        "--time_increment",
        type=int,
        default=1,
        metavar="N",
        help="time_increment for moving window",
    )

    dataset_group.add_argument(
        "--shuffle_train_val",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="shuffle the available train and val samples to avoid distribution shift issues. This shuffling occurs between the train and val samples (obtained from the standard sequential train/val/test splits), some train samples are used for validation, some val samples used for training.",
    )

    # dataset_group.add_argument(
    #     "--noaug", default=False, action="store_true", help="no data augmentation"
    # )

    # TODO: change this to aug
    dataset_group.add_argument(
        "--noaug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="no data augmentation",
    )

    # TODO: what exactly is this doing again?
    dataset_group.add_argument(
        "--label_corrupt_prob",
        type=float,
        default=0.0,
        metavar="PROB",
        help="probability of corrupting labels for robustness testing (0.0-1.0)",
    )

    dataset_group.add_argument(
        "--plot_dataset",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Plot the used dataset",
    )

    # parser.add_argument(
    #     "--trainloader", default="", help="path to the dataloader with random labels"
    # )
    # parser.add_argument(
    #     "--testloader", default="", help="path to the testloader with random labels"
    # )

    # Experiment
    exp_group = parser.add_argument_group("Experiment", "Experiment settings")

    # TODO: should also enable attention matrix plotting etc
    exp_group.add_argument(
        "--enable_plotting",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="enable test result plotting",
    )

    return parser


# Additional arguments


def _add_time_series_forecast_args(parser):
    """Add time series forecasting-specific arguments to Experiment group."""
    # Add to existing Experiment group
    exp_group = _get_argument_group(parser, "Experiment")
    if exp_group:
        # seq_len denotes input history length, horizon denotes output future length
        exp_group.add_argument(
            "--seq_len",
            type=int,
            default=512,
            metavar="N",
            help="input sequence length (history length)",
        )
        exp_group.add_argument(
            "--pred_len",
            type=int,
            default=96,
            metavar="N",
            help="output prediction horizon (future length)",
        )
        exp_group.add_argument(
            "--input_channels",
            type=int,
            default=None,
            metavar="N",
            help="number of input channels a.k.a encoder input size",
        )

    return parser


def _add_deep_learning_args(parser):
    """Extended configuration parser including deep learning model arguments."""

    # Hardware
    hw_group = parser.add_argument_group("Hardware", "Hardware settings")
    hw_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        metavar="DEV",
        help="device to use for training (cpu or cuda:0, cuda:1, etc.)",
    )

    deep_learning_group = parser.add_argument_group("Deep Learning", "Deep Learning settings")

    # Optimizer
    deep_learning_group.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        metavar="OPT",
        help="Optimizer to use (e.g., Adam, SGD, AdamW). Use same case as class names in torch.optim",
    )

    # Hyperparameters
    deep_learning_group.add_argument(
        "--batch_size",
        type=int,
        default=256,
        metavar="N",
        help="batch size for training",
    )
    deep_learning_group.add_argument(
        "--lrate",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate",
    )
    # TODO: check help here again
    deep_learning_group.add_argument(
        "--wdecay",
        type=float,
        default=1e-5,
        metavar="WD",
        help="weight decay (L2 regularization)",
    )
    deep_learning_group.add_argument(
        "--clip_grad_value",
        type=float,
        default=0,
        metavar="VAL",
        help="gradient clipping value (0 = no clipping)",
    )

    deep_learning_group.add_argument(
        "--use_revin",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use reversible instance normalization",
    )

    deep_learning_group.add_argument(
        "--revin_affine",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use affine transformation in RevIN",
    )

    deep_learning_group.add_argument(
        "--revin_subtract_last",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="False: subtract mean; True: subtract last value in RevIN",
    )

    # Add to existing Experiment group
    exp_group = _get_argument_group(parser, "Experiment")
    if exp_group:
        exp_group.add_argument(
            "--max_epochs",
            type=int,
            default=100,
            metavar="N",
            help="maximum number of training epochs",
        )
        exp_group.add_argument(
            "--patience",
            type=int,
            default=1,
            metavar="N",
            help="patience for early stopping (number of epochs without improvement)",
        )

    post_training_analysis_group = parser.add_argument_group(
        "Post training analysis", "Post training analysis settings"
    )

    # Post training analysis
    post_training_analysis_group.add_argument(
        "--hessian_analysis",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="calculate and plot hessian values",
    )

    return parser


def _add_scheduler_args(parser):
    """Add learning rate scheduler arguments."""
    scheduler_group = parser.add_argument_group("LR Scheduler", "Learning rate scheduler settings")

    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        default="cosine_warm_restarts",
        choices=["cosine_warm_restarts", "linear", "none"],
        metavar="SCHEDULER",
        help="learning rate scheduler: cosine_warm_restarts | linear | none",
    )
    # CosineAnnealingWarmRestarts parameters
    scheduler_group.add_argument(
        "--scheduler_T0",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs for first restart (CosineAnnealingWarmRestarts)",
    )
    scheduler_group.add_argument(
        "--scheduler_T_mult",
        type=int,
        default=1,
        metavar="N",
        help="factor to increase T_i after each restart (CosineAnnealingWarmRestarts)",
    )
    scheduler_group.add_argument(
        "--scheduler_eta_min",
        type=float,
        default=1e-6,
        metavar="LR",
        help="minimum learning rate",
    )

    return parser


# def _add_loss_landscape_args(parser):
#     """Extended configuration parser including loss landscape and plotting arguments."""
#
#     # Loss landscape group with description
#     ll_group = parser.add_argument_group(
#         "Loss Landscape Visualization",
#         "Loss Landscape Visualization settings,\n For more details, visit: https://github.com/tomgoldstein/loss-landscape",
#     )
#     ll_group.add_argument(
#         "--plot_surface_mpi",
#         "-m",
#         action="store_true",
#         help="use mpi for loss landscape computation",
#     )
#     ll_group.add_argument(
#         "--plot_surface_cuda",
#         "-c",
#         action="store_true",
#         help="use cuda for loss landscape computation",
#     )
#     ll_group.add_argument(
#         "--threads",
#         default=2,
#         type=int,
#         metavar="N",
#         help="number of threads",
#     )
#     ll_group.add_argument(
#         "--ngpu",
#         type=int,
#         default=1,
#         metavar="N",
#         help="number of GPUs to use for each rank, useful for data parallel evaluation",
#     )
#
#     # Model parameters group
#     model_params_group = parser.add_argument_group(
#         "Loss Landscape Visualization", "Model Parameters"
#     )
#     model_params_group.add_argument(
#         "--model",
#         default="samformer",
#         metavar="NAME",
#         help="model name",
#     )
#     model_params_group.add_argument(
#         "--model_file",
#         default="experiments/samformer/ETTh1/final_model_s2024.pt",
#         metavar="PATH",
#         help="path to the trained model file",
#     )
#     model_params_group.add_argument(
#         "--model_file2",
#         default="",
#         metavar="PATH",
#         help="use (model_file2 - model_file) as the xdirection",
#     )
#     model_params_group.add_argument(
#         "--model_file3",
#         default="",
#         metavar="PATH",
#         help="use (model_file3 - model_file) as the ydirection",
#     )
#
#     # Direction parameters group
#     dir_group = parser.add_argument_group(
#         "Loss Landscape Visualization",
#         "Direction Parameters",
#     )
#     dir_group.add_argument(
#         "--dir_file",
#         default="",
#         metavar="PATH",
#         help="specify the name of direction file, or the path to an existing direction file",
#     )
#     dir_group.add_argument(
#         "--dir_type",
#         default="weights",
#         metavar="TYPE",
#         help="direction type: weights | states (including BN's running_mean/var)",
#     )
#     dir_group.add_argument(
#         "--x",
#         default="-1:1:51",
#         metavar="MIN:MAX:NUM",
#         help="A string with format xmin:x_max:xnum",
#     )
#     dir_group.add_argument(
#         "--y",
#         default=None,
#         metavar="MIN:MAX:NUM",
#         help="A string with format ymin:ymax:ynum",
#     )
#     dir_group.add_argument(
#         "--xnorm",
#         default="",
#         metavar="TYPE",
#         help="direction normalization: filter | layer | weight",
#     )
#     dir_group.add_argument(
#         "--ynorm",
#         default="",
#         metavar="TYPE",
#         help="direction normalization: filter | layer | weight",
#     )
#     dir_group.add_argument(
#         "--xignore",
#         default="",
#         metavar="TYPE",
#         help="ignore bias and BN parameters: biasbn",
#     )
#     dir_group.add_argument(
#         "--yignore",
#         default="",
#         metavar="TYPE",
#         help="ignore bias and BN parameters: biasbn",
#     )
#     dir_group.add_argument(
#         "--same_dir",
#         action="store_true",
#         default=False,
#         help="use the same random direction for both x-axis and y-axis",
#     )
#     dir_group.add_argument(
#         "--idx",
#         default=0,
#         type=int,
#         metavar="N",
#         help="the index for the repeatness experiment",
#     )
#     dir_group.add_argument(
#         "--surf_file",
#         default="",
#         metavar="PATH",
#         help="customize the name of surface file, could be an existing file.",
#     )
#     dir_group.add_argument(
#         "--hessian_directions",
#         action="store_true",
#         default=False,
#         help="create hessian eigenvectors directions h5 file",
#     )
#
#     # Plot parameters group
#     plot_group = parser.add_argument_group("Loss Landscape Visualization", "Plot Parameters")
#     plot_group.add_argument(
#         "--proj_file",
#         default="",
#         metavar="PATH",
#         help="the .h5 file contains projected optimization trajectory.",
#     )
#     plot_group.add_argument(
#         "--loss_max",
#         default=5,
#         type=float,
#         metavar="VAL",
#         help="Maximum value to show in 1D plot",
#     )
#     plot_group.add_argument(
#         "--vmax",
#         default=10,
#         type=float,
#         metavar="VAL",
#         help="Maximum value to map",
#     )
#     plot_group.add_argument(
#         "--vmin",
#         default=0.1,
#         type=float,
#         metavar="VAL",
#         help="Minimum value to map",
#     )
#     plot_group.add_argument(
#         "--vlevel",
#         default=0.5,
#         type=float,
#         metavar="VAL",
#         help="plot contours every vlevel",
#     )
#     plot_group.add_argument(
#         "--show", action="store_true", default=False, help="show plotted figures"
#     )
#     plot_group.add_argument(
#         "--log",
#         action="store_true",
#         default=False,
#         help="use log scale for loss values",
#     )
#     plot_group.add_argument(
#         "--plot",
#         action="store_true",
#         default=False,
#         help="plot figures after computation",
#     )
#
#     return parser


def _add_sam_args(parser):
    """Add SAM (Sharpness Aware Minimization) arguments."""
    sam_group = parser.add_argument_group(
        "SAM Optimization", "Sharpness Aware Minimization settings"
    )

    sam_group.add_argument(
        "--sam",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use Sharpness Aware Minimization",
    )

    sam_group.add_argument(
        "--rho",
        type=float,
        default=0.5,
        metavar="RHO",
        help="neighborhood size for SAM",
    )

    sam_group.add_argument(
        "--sam_adaptive",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use Adaptive SAM (ASAM)",
    )

    return parser


def _add_gsam_args(parser):
    # Default values taken from:
    # https://github.com/juntang-zhuang/GSAM/blob/main/gsam/gsam.py
    # https://github.com/juntang-zhuang/GSAM/blob/main/example/train.py
    """Add GSAM (Gradient-based SAM) arguments."""
    gsam_group = parser.add_argument_group(
        "GSAM Optimization",
        "Gradient-based Sharpness Aware Minimization settings. Default values taken from: https://github.com/juntang-zhuang/GSAM/blob/main/gsam/gsam.py, and https://github.com/juntang-zhuang/GSAM/blob/main/example/train.py",
    )

    gsam_group.add_argument(
        "--gsam",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use GSAM (Gradient-based SAM) instead of standard SAM",
    )
    gsam_group.add_argument(
        "--gsam_alpha",
        type=float,
        default=0.4,
        metavar="ALPHA",
        help="alpha parameter for GSAM",
    )
    gsam_group.add_argument(
        "--gsam_rho_max",
        type=float,
        default=0.5,
        metavar="RHO_MAX",
        help="maximum rho value for GSAM",
    )
    gsam_group.add_argument(
        "--gsam_rho_min",
        type=float,
        default=0.1,
        metavar="RHO_MIN",
        help="minimum rho value for GSAM",
    )
    gsam_group.add_argument(
        "--gsam_adaptive",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use adaptive GSAM (I think this uses adaptive SAM with the GSAM algo)",
    )
    return parser


def _add_fsam_args(parser):
    # Default values taken from: https://github.com/nblt/F-SAM/blob/main/train.py
    """Add FSAM (Friendly SAM) arguments."""
    fsam_group = parser.add_argument_group(
        "FSAM Optimization",
        "Friendly Sharpness Aware Minimization settings. Default values taken from https://github.com/nblt/F-SAM/blob/main/train.py",
    )

    fsam_group.add_argument(
        "--fsam",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use Friendly Sharpness Aware Minimization (FriendlySAM)",
    )
    fsam_group.add_argument(
        "--fsam_rho",
        type=float,
        default=0.05,
        metavar="RHO",
        help="neighborhood size (rho) for FSAM perturbation",
    )
    fsam_group.add_argument(
        "--fsam_sigma",
        type=float,
        default=1.0,
        metavar="SIGMA",
        help="scaling factor for momentum subtraction in FSAM",
    )
    fsam_group.add_argument(
        "--fsam_lmbda",
        type=float,
        default=0.95,
        metavar="LAMBDA",
        help="momentum decay factor (lambda) for FSAM",
    )
    fsam_group.add_argument(
        "--fsam_adaptive",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use adaptive FSAM (I think this uses adaptive SAM with the FSAM algo)",
    )
    return parser


# Model specific arguments


def _add_samformer_args(parser):
    """Add SAMFormer model architecture arguments."""

    # SAMFormer specific arguments
    samformer_group = parser.add_argument_group(
        "SAMFormer Model", "SAMFormer-specific model architecture hyperparameters"
    )

    samformer_group.add_argument(
        "--hid_dim", type=int, default=16, metavar="N", help="hidden dimension size"
    )

    # TODO: maybe not needed
    samformer_group.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["gelu", "relu", "swish"],
        metavar="ACTIVATION",
        help="activation function",
    )

    samformer_group.add_argument(
        "--attn_dropout",
        type=float,
        default=0,
        metavar="RATE",
        help="dropout rate for SAMFormer attention layers",
    )

    samformer_group.add_argument(
        "--output_dropout",
        type=float,
        default=0,
        metavar="RATE",
        help="dropout rate for SAMFormer output layers",
    )

    return parser


def _add_tsmixer_args(parser):
    """Add TSMixer model architecture arguments."""

    tsmixer_group = parser.add_argument_group(
        "TSMixer Model", "TSMixer-specific model architecture hyperparameters"
    )

    tsmixer_group.add_argument(
        "--output_channels",
        type=int,
        default=None,
        metavar="N",
        help="number of output channels",
    )

    tsmixer_group.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["gelu", "relu", "swish"],
        metavar="ACTIVATION",
        help="activation function",
    )
    tsmixer_group.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        metavar="N",
        help="number of mixer blocks",
    )
    tsmixer_group.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        metavar="RATE",
        help="dropout rate for TSMixer",
    )
    tsmixer_group.add_argument(
        "--ff_dim",
        type=int,
        default=64,
        metavar="N",
        help="feedforward dimension in mixer layers",
    )
    tsmixer_group.add_argument(
        "--normalize_before",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="whether to normalize before mixer layers",
    )
    tsmixer_group.add_argument(
        "--norm_type",
        type=str,
        default="batch",
        choices=["batch", "layer"],
        metavar="TYPE",
        help="type of normalization",
    )

    return parser


def _add_tsmixer_ext_args(parser):
    """Add TSMixerExt-specific arguments (in addition to TSMixer args)."""

    tsmixer_ext_group = parser.add_argument_group(
        "TSMixerExt Model", "TSMixerExt-specific model architecture hyperparameters"
    )

    tsmixer_ext_group.add_argument(
        "--extra_channels",
        type=int,
        default=None,
        metavar="N",
        help="number of extra (future known) input channels",
    )
    tsmixer_ext_group.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        metavar="N",
        help="number of hidden channels in mixer layers",
    )
    tsmixer_ext_group.add_argument(
        "--static_channels",
        type=int,
        default=1,
        metavar="N",
        help="number of static feature channels",
    )

    return parser


def _add_formers_common_args(parser):
    """Add common Transformer architecture arguments shared by Formers and PatchTST."""

    formers_common_group = parser.add_argument_group(
        "Transformer Architecture (Common)",
        "Common architecture hyperparameters shared by Formers and PatchTST models",
    )

    # Core architecture dimensions
    formers_common_group.add_argument(
        "--c_out",
        type=int,
        default=7,
        metavar="N",
        help="output size (number of output channels)",
    )
    formers_common_group.add_argument(
        "--d_model",
        type=int,
        default=512,
        metavar="N",
        help="Transformer dimension of model",
    )
    formers_common_group.add_argument(
        "--n_heads",
        type=int,
        default=8,
        metavar="N",
        help="number of attention heads",
    )
    formers_common_group.add_argument(
        "--e_layers",
        type=int,
        default=2,
        metavar="N",
        help="number of encoder layers",
    )
    formers_common_group.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        metavar="N",
        help="dimension of feed-forward network (Transfomer MLP dimension)",
    )

    # Regularization
    formers_common_group.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        metavar="RATE",
        help="dropout rate",
    )
    formers_common_group.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=["gelu", "relu", "swish"],
        metavar="",
        help="activation function",
    )

    return parser


def _add_formers_specific_args(parser):
    """Add Formers (Autoformer, Informer, Transformer) specific arguments."""

    formers_specific_group = parser.add_argument_group(
        "Formers Models",
        "Formers-specific architecture hyperparameters (Autoformer, Informer, Transformer)",
    )

    # Decoder architecture
    formers_specific_group.add_argument(
        "--dec_in",
        type=int,
        default=7,
        metavar="N",
        help="decoder input size",
    )
    formers_specific_group.add_argument(
        "--d_layers",
        type=int,
        default=1,
        metavar="N",
        help="number of decoder layers",
    )

    # Forecasting task parameters
    formers_specific_group.add_argument(
        "--label_len",
        type=int,
        default=48,
        metavar="N",
        help="start token length for decoder",
    )

    # TODO: do we need this?
    # Embedding
    formers_specific_group.add_argument(
        "--embed_type",
        type=int,
        default=0,
        metavar="TYPE",
        help="time features encoding: 0=default, 1=value+temporal+positional, 2=value+temporal, 3=value+positional, 4=value only",
    )
    formers_specific_group.add_argument(
        "--embed",
        type=str,
        default="timeF",
        choices=["timeF", "fixed", "learned"],
        metavar="TYPE",
        help="time features encoding method",
    )

    # TODO: maybe move to dataloader, maybe we already have a similar arg
    formers_specific_group.add_argument(
        "--freq",
        type=str,
        default="h",
        metavar="FREQ",
        help="frequency for time features encoding: s=secondly, t=minutely, h=hourly, d=daily, b=business days, w=weekly, m=monthly",
    )

    # Time features flag
    formers_specific_group.add_argument(
        "--use_time_features",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use time features encoding for Transformer models",
    )

    # TODO: maybe move to dataloader, maybe we already have a similar arg
    # Model-specific parameters
    formers_specific_group.add_argument(
        "--moving_avg",
        type=int,
        default=25,
        metavar="N",
        help="window size of moving average (for Autoformer)",
    )
    formers_specific_group.add_argument(
        "--factor",
        type=int,
        default=5,
        metavar="N",
        help="attention factor (for Informer ProbSparse attention)",
    )
    formers_specific_group.add_argument(
        "--attn",
        type=str,
        default="prob",
        choices=["prob", "full"],
        metavar="TYPE",
        help="attention type: prob (ProbSparse) or full (for Informer)",
    )
    formers_specific_group.add_argument(
        "--distil",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="whether to use distilling in encoder (for Informer)",
    )
    formers_specific_group.add_argument(
        "--mix",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="whether to use mix attention in decoder (for Informer)",
    )
    # TODO: merge with existing "plot_attention" functionality that already has
    # a logic to save attention weights
    formers_specific_group.add_argument(
        "--output_attention",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether to output attention weights in encoder",
    )

    # Prediction mode
    formers_specific_group.add_argument(
        "--do_predict",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether to predict unseen future data",
    )

    return parser


def _add_itransformer_args(parser):
    """Add iTransformer-specific arguments."""

    itransformer_group = parser.add_argument_group(
        "iTransformer Model", "iTransformer-specific model architecture hyperparameters"
    )

    itransformer_group.add_argument(
        "--use_norm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use normalization from Non-stationary Transformer",
    )
    itransformer_group.add_argument(
        "--class_strategy",
        type=str,
        default="projection",
        choices=["projection", "average", "cls_token"],
        metavar="STRATEGY",
        help="classification strategy (for classification tasks)",
    )
    itransformer_group.add_argument(
        "--factor",
        type=int,
        default=5,
        metavar="N",
        help="attention factor",
    )
    itransformer_group.add_argument(
        "--embed",
        type=str,
        default="timeF",
        choices=["timeF", "fixed", "learned"],
        metavar="TYPE",
        help="time features encoding method",
    )
    itransformer_group.add_argument(
        "--freq",
        type=str,
        default="h",
        metavar="FREQ",
        help="frequency for time features encoding: s=secondly, t=minutely, h=hourly, d=daily, b=business days, w=weekly, m=monthly",
    )
    itransformer_group.add_argument(
        "--use_time_features",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use time features encoding",
    )
    itransformer_group.add_argument(
        "--output_attention",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether to output attention weights in encoder",
    )

    return parser


def _add_dlinear_args(parser):
    """Add DLinear model architecture arguments."""

    dlinear_group = parser.add_argument_group(
        "DLinear Model", "DLinear-specific model architecture hyperparameters"
    )

    dlinear_group.add_argument(
        "--individual",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use individual linear layers for each channel",
    )
    dlinear_group.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        metavar="N",
        help="kernel size for moving average decomposition",
    )

    return parser


def _add_fedformer_args(parser):
    """Add FEDformer-specific arguments."""

    fedformer_group = parser.add_argument_group(
        "FEDformer Model", "FEDformer-specific model architecture hyperparameters"
    )

    fedformer_group.add_argument(
        "--version",
        type=str,
        default="Fourier",
        choices=["Fourier", "Wavelets"],
        metavar="VERSION",
        help="FEDformer version: Fourier or Wavelets",
    )

    fedformer_group.add_argument(
        "--mode_select",
        type=str,
        default="random",
        choices=["random", "low"],
        metavar="MODE",
        help="mode selection method for Fourier: random or low frequency",
    )
    fedformer_group.add_argument(
        "--modes",
        type=int,
        default=32,
        metavar="N",
        help="number of Fourier/Wavelet modes to use",
    )
    # Wavelet-specific parameters
    fedformer_group.add_argument(
        "--wavelet_L",
        type=int,
        default=1,
        metavar="N",
        help="number of levels for wavelet decomposition",
    )
    fedformer_group.add_argument(
        "--wavelet_base",
        type=str,
        default="legendre",
        choices=["legendre", "chebyshev"],
        metavar="BASE",
        help="base function for wavelet transform",
    )
    fedformer_group.add_argument(
        "--cross_activation",
        type=str,
        default="tanh",
        choices=["tanh", "softmax"],
        metavar="ACTIVATION",
        help="activation function for cross attention in wavelet version",
    )

    return parser


def _add_linear_args(parser):
    """Add Linear model architecture arguments."""

    # Linear model uses enc_in from _add_time_series_forecast_args
    # No additional arguments needed, but keeping function for consistency
    # and potential future model-specific args

    return parser


def _add_nlinear_args(parser):
    """Add NLinear model architecture arguments."""

    # NLinear model uses enc_in from _add_time_series_forecast_args
    # No additional arguments needed, but keeping function for consistency
    # and potential future model-specific args

    return parser


def _add_patchtst_specific_args(parser):
    """Add PatchTST-specific arguments (after loading common Transformer args)."""

    patchtst_specific_group = parser.add_argument_group(
        "PatchTST Specific", "PatchTST-specific architecture hyperparameters"
    )

    patchtst_specific_group.add_argument(
        "--d_k",
        type=int,
        default=None,
        metavar="N",
        help="dimension of keys (default: d_model // n_heads)",
    )
    patchtst_specific_group.add_argument(
        "--d_v",
        type=int,
        default=None,
        metavar="N",
        help="dimension of values (default: d_model // n_heads)",
    )

    # PatchTST-specific dropout
    patchtst_specific_group.add_argument(
        "--fc_dropout",
        type=float,
        default=0.05,
        metavar="RATE",
        help="fully connected dropout",
    )
    patchtst_specific_group.add_argument(
        "--head_dropout",
        type=float,
        default=0.0,
        metavar="RATE",
        help="head dropout",
    )
    patchtst_specific_group.add_argument(
        "--attn_dropout",
        type=float,
        default=0.0,
        metavar="RATE",
        help="attention dropout",
    )

    # Patch parameters
    patchtst_specific_group.add_argument(
        "--patch_len",
        type=int,
        default=16,
        metavar="N",
        help="patch length",
    )
    patchtst_specific_group.add_argument(
        "--stride",
        type=int,
        default=8,
        metavar="N",
        help="stride for patch creation",
    )
    patchtst_specific_group.add_argument(
        "--padding_patch",
        type=str,
        default="end",
        choices=["end", "none"],
        metavar="TYPE",
        help="padding type for patches",
    )

    # Normalization - PatchTST-specific RevIN parameters
    # patchtst_specific_group.add_argument(
    #     "--revin",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=True,
    #     help="use reversible instance normalization (RevIN)",
    # )
    # patchtst_specific_group.add_argument(
    #     "--affine",
    #     type=str2bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="use affine transformation in RevIN",
    # )

    # Decomposition
    patchtst_specific_group.add_argument(
        "--decomposition",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use series decomposition",
    )
    patchtst_specific_group.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        metavar="N",
        help="decomposition kernel size",
    )

    # Model architecture options
    patchtst_specific_group.add_argument(
        "--individual",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="individual head for each channel",
    )

    # Normalization type
    patchtst_specific_group.add_argument(
        "--norm",
        type=str,
        default="BatchNorm",
        choices=["BatchNorm", "LayerNorm"],
        metavar="TYPE",
        help="normalization type",
    )

    # Attention mechanism
    patchtst_specific_group.add_argument(
        "--res_attention",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use residual attention",
    )
    patchtst_specific_group.add_argument(
        "--pre_norm",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="apply normalization before attention (pre-norm vs post-norm)",
    )
    patchtst_specific_group.add_argument(
        "--store_attn",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="store attention weights",
    )

    # Positional encoding
    patchtst_specific_group.add_argument(
        "--pe",
        type=str,
        default="zeros",
        choices=["zeros", "normal", "uniform"],
        metavar="TYPE",
        help="positional encoding initialization",
    )
    patchtst_specific_group.add_argument(
        "--learn_pe",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="learn positional encoding",
    )

    # Head configuration
    patchtst_specific_group.add_argument(
        "--pretrain_head",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use pretrained head",
    )
    patchtst_specific_group.add_argument(
        "--head_type",
        type=str,
        default="flatten",
        choices=["flatten", "prediction"],
        metavar="TYPE",
        help="head type",
    )

    # Legacy parameters (kept for compatibility)
    patchtst_specific_group.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        metavar="N",
        help="maximum sequence length (legacy parameter, kept for compatibility)",
    )
    patchtst_specific_group.add_argument(
        "--key_padding_mask",
        type=str,
        default="auto",
        metavar="TYPE",
        help="key padding mask type (legacy parameter, kept for compatibility)",
    )
    patchtst_specific_group.add_argument(
        "--padding_var",
        type=int,
        default=None,
        metavar="N",
        help="padding variable (legacy parameter, kept for compatibility)",
    )
    patchtst_specific_group.add_argument(
        "--attn_mask",
        type=str,
        default=None,
        metavar="TYPE",
        help="attention mask (legacy parameter, kept for compatibility)",
    )
    patchtst_specific_group.add_argument(
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="verbose output (legacy parameter, kept for compatibility)",
    )

    return parser


def _add_moment_args(parser):
    """Add MOMENT model architecture arguments."""

    moment_group = parser.add_argument_group(
        "MOMENT Model", "MOMENT foundation model hyperparameters"
    )

    moment_group.add_argument(
        "--moment_model_name",
        type=str,
        default="AutonLab/MOMENT-1-large",
        metavar="NAME",
        help="MOMENT model name from HuggingFace (e.g., AutonLab/MOMENT-1-large)",
    )
    moment_group.add_argument(
        "--head_dropout",
        type=float,
        default=0.1,
        metavar="RATE",
        help="dropout rate for forecasting head",
    )
    moment_group.add_argument(
        "--freeze_encoder",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="freeze the transformer encoder",
    )
    moment_group.add_argument(
        "--freeze_embedder",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="freeze the patch embedding layer",
    )
    moment_group.add_argument(
        "--freeze_head",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="freeze the forecasting head (usually False for training)",
    )
    moment_group.add_argument(
        "--log_interval",
        type=int,
        default=50,
        metavar="N",
        help="log detailed metrics every N batches during training",
    )

    return parser


def _add_timesfm_args(parser):
    """Add TimesFM model arguments."""
    timesfm_group = parser.add_argument_group(
        "TimesFM Model", "TimesFM foundation model hyperparameters"
    )

    timesfm_group.add_argument(
        "--timesfm_model_name",
        type=str,
        default="google/timesfm-2.5-200m-pytorch",
        metavar="NAME",
        help="TimesFM model name from HuggingFace",
    )
    timesfm_group.add_argument(
        "--timesfm_backend",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        metavar="BACKEND",
        help="Backend for TimesFM inference",
    )
    timesfm_group.add_argument(
        "--timesfm_per_core_batch_size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size per core for TimesFM inference",
    )

    return parser


def _add_timekan_args(parser):
    """Add TimeKAN model architecture arguments."""

    timekan_group = parser.add_argument_group(
        "TimeKAN Model", "TimeKAN-specific model architecture hyperparameters"
    )

    timekan_group.add_argument(
        "--d_model",
        type=int,
        default=32,
        metavar="N",
        help="model dimension for TimeKAN",
    )
    timekan_group.add_argument(
        "--e_layers",
        type=int,
        default=2,
        metavar="N",
        help="number of encoder layers",
    )
    timekan_group.add_argument(
        "--down_sampling_layers",
        type=int,
        default=2,
        metavar="N",
        help="number of downsampling layers for multi-scale processing",
    )
    timekan_group.add_argument(
        "--down_sampling_window",
        type=int,
        default=2,
        metavar="N",
        help="downsampling window size (seq_len must be >= window^layers)",
    )
    timekan_group.add_argument(
        "--moving_avg",
        type=int,
        default=25,
        metavar="N",
        help="moving average kernel size for series decomposition",
    )
    timekan_group.add_argument(
        "--embed",
        type=str,
        default="timeF",
        choices=["timeF", "fixed", "learned"],
        metavar="TYPE",
        help="time features encoding method",
    )
    timekan_group.add_argument(
        "--freq",
        type=str,
        default="h",
        metavar="FREQ",
        help="frequency for time features encoding",
    )
    timekan_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        metavar="RATE",
        help="dropout rate",
    )
    timekan_group.add_argument(
        "--use_norm",
        type=int,
        default=1,
        choices=[0, 1],
        metavar="N",
        help="whether to use normalization (0=no, 1=yes)",
    )
    timekan_group.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        choices=[0, 1],
        metavar="N",
        help="channel independence flag (0=dependent, 1=independent)",
    )
    timekan_group.add_argument(
        "--begin_order",
        type=int,
        default=2,
        metavar="N",
        help="beginning order for Chebyshev polynomials in KAN layers",
    )
    timekan_group.add_argument(
        "--use_future_temporal_feature",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use future temporal features",
    )
    timekan_group.add_argument(
        "--c_out",
        type=int,
        default=None,
        metavar="N",
        help="number of output channels (default: same as input_channels)",
    )
    timekan_group.add_argument(
        "--label_len",
        type=int,
        default=0,
        metavar="N",
        help="decoder start token length",
    )
    timekan_group.add_argument(
        "--use_time_features",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="use time features encoding for TimeKAN embedding",
    )

    return parser


# def _add_tritracknet_args(parser):
#     """Add TriTrackNet model architecture arguments."""
#
#     tritracknet_group = parser.add_argument_group(
#         "TriTrackNet Model", "TriTrackNet-specific model architecture hyperparameters"
#     )
#
#     tritracknet_group.add_argument(
#         "--hid_dim",
#         type=int,
#         default=16,
#         metavar="N",
#         help="hidden dimension size for attention",
#     )
#     tritracknet_group.add_argument(
#         "--mlp_hidden_dim",
#         type=int,
#         default=64,
#         metavar="N",
#         help="hidden dimension size for MLP in HFF",
#     )
#     tritracknet_group.add_argument(
#         "--cross_embed_dim",
#         type=int,
#         default=128,
#         metavar="N",
#         help="embedding dimension for cross-modal fusion",
#     )
#     tritracknet_group.add_argument(
#         "--knowledge_input_dim",
#         type=int,
#         default=768,
#         metavar="N",
#         help="dimension of external knowledge embeddings (e.g., 768 for BERT)",
#     )
#     tritracknet_group.add_argument(
#         "--use_aux",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=False,
#         help="enable cross-modal fusion with external knowledge",
#     )
#     tritracknet_group.add_argument(
#         "--use_attention",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=True,
#         help="use standard attention in dual-channel attention",
#     )
#     tritracknet_group.add_argument(
#         "--use_hff",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=True,
#         help="use heterogeneous feature fusion (HFF)",
#     )
#     tritracknet_group.add_argument(
#         "--aux_mode",
#         type=str,
#         default="keep",
#         choices=["keep", "off"],
#         metavar="MODE",
#         help="auxiliary mode for reverse attention: keep | off",
#     )
#     tritracknet_group.add_argument(
#         "--use_gating",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=True,
#         help="use gating mechanism in cross-modal fusion",
#     )
#     tritracknet_group.add_argument(
#         "--gating_mode",
#         type=str,
#         default="adaptive",
#         choices=["adaptive", "ones", "average"],
#         metavar="MODE",
#         help="gating mode: adaptive | ones | average",
#     )
#
#     return parser


# def _add_timecf_args(parser):
#     """Add TimeCF model architecture arguments."""
#
#     timecf_group = parser.add_argument_group(
#         "TimeCF Model", "TimeCF-specific model architecture hyperparameters"
#     )
#
#     timecf_group.add_argument(
#         "--d_model",
#         type=int,
#         default=32,
#         metavar="N",
#         help="Model embedding dimension",
#     )
#     timecf_group.add_argument(
#         "--d_ff",
#         type=int,
#         default=64,
#         metavar="N",
#         help="Feed-forward network dimension",
#     )
#     timecf_group.add_argument(
#         "--num_layers",
#         type=int,
#         default=2,
#         metavar="N",
#         help="Number of PDMC layers",
#     )
#     timecf_group.add_argument(
#         "--num_scales",
#         type=int,
#         default=4,
#         metavar="N",
#         help="Number of multi-scale levels",
#     )
#     timecf_group.add_argument(
#         "--downsample_rate",
#         type=int,
#         default=2,
#         metavar="N",
#         help="Downsampling rate between scales",
#     )
#     timecf_group.add_argument(
#         "--decomp_kernel_size",
#         type=int,
#         default=25,
#         metavar="N",
#         help="Kernel size for series decomposition",
#     )
#     timecf_group.add_argument(
#         "--conv_kernel_size",
#         type=int,
#         default=3,
#         metavar="N",
#         help="Kernel size for convolution blocks",
#     )
#     timecf_group.add_argument(
#         "--dropout",
#         type=float,
#         default=0.1,
#         metavar="RATE",
#         help="Dropout rate",
#     )
#     timecf_group.add_argument(
#         "--conv_alpha",
#         type=float,
#         default=0.5,
#         metavar="ALPHA",
#         help="Alpha weight for convolution residual",
#     )
#
#     # SAMFre parameters
#     timecf_group.add_argument(
#         "--use_samfre_loss",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=True,
#         help="Use SAMFre loss",
#     )
#     timecf_group.add_argument(
#         "--samfre_alpha",
#         type=float,
#         default=0.5,
#         metavar="ALPHA",
#         help="Alpha for SAMFre loss (0=MSE, 1=FFT)",
#     )
#     timecf_group.add_argument(
#         "--sam_start_updates",
#         type=int,
#         default=1000,
#         metavar="N",
#         help="Number of updates before activating SAM",
#     )
#
#     # Temporal embedding
#     timecf_group.add_argument(
#         "--use_temporal_embedding",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=False,
#         help="Use temporal embedding (time features)",
#     )
#     timecf_group.add_argument(
#         "--freq",
#         type=str,
#         default="h",
#         metavar="FREQ",
#         help="Frequency for temporal features (h=hourly)",
#     )
#
#     return parser


def _add_statsforecast_args(parser):
    """Add StatsForecast arguments."""
    statsforecast_group = parser.add_argument_group(
        "StatsForecast", "StatsForecast-specific settings"
    )

    statsforecast_group.add_argument(
        "--freq",
        type=str,
        default="h",
        metavar="FREQ",
        help="frequency of the data (see pandas available frequencies, e.g., 'h' for hourly, 'D' for daily)",
    )
    statsforecast_group.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        metavar="N",
        help="number of jobs used in parallel processing (-1 for all cores, use only for for expensive models like AutoARIMA, where computation time dominates overhead)",
    )

    return parser


def _add_auto_arima_args(parser):
    """Add Auto ARIMA arguments."""
    auto_arima_group = parser.add_argument_group("Auto ARIMA", "Auto ARIMA model-specific settings")

    # StatsForecast specific parameters
    # TODO: change this some auto value as in the code down below
    auto_arima_group.add_argument(
        "--n_cores",
        type=int,
        default=8,
        metavar="N",
        help="number of cores for parallel processing",
    )

    # ARIMA specific parameters
    auto_arima_group.add_argument(
        "--seasonal",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use seasonal ARIMA",
    )
    auto_arima_group.add_argument(
        "--seasonal_periods",
        type=int,
        default=24,
        metavar="N",
        help="seasonal periods",
    )
    auto_arima_group.add_argument(
        "--auto_arima",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use auto ARIMA parameter selection",
    )
    auto_arima_group.add_argument(
        "--p",
        type=int,
        default=1,
        metavar="N",
        help="autoregressive order",
    )
    auto_arima_group.add_argument(
        "--d",
        type=int,
        default=1,
        metavar="N",
        help="differencing order",
    )
    auto_arima_group.add_argument(
        "--q",
        type=int,
        default=1,
        metavar="N",
        help="moving average order",
    )
    auto_arima_group.add_argument(
        "--P",
        type=int,
        default=1,
        metavar="N",
        help="seasonal AR order",
    )
    auto_arima_group.add_argument(
        "--D",
        type=int,
        default=1,
        metavar="N",
        help="seasonal differencing order",
    )
    auto_arima_group.add_argument(
        "--Q",
        type=int,
        default=1,
        metavar="N",
        help="seasonal MA order",
    )
    auto_arima_group.add_argument(
        "--max_p",
        type=int,
        default=3,
        metavar="N",
        help="maximum AR order",
    )
    auto_arima_group.add_argument(
        "--max_q",
        type=int,
        default=3,
        metavar="N",
        help="maximum MA order",
    )
    auto_arima_group.add_argument(
        "--max_d",
        type=int,
        default=2,
        metavar="N",
        help="maximum differencing order",
    )
    auto_arima_group.add_argument(
        "--max_P",
        type=int,
        default=2,
        metavar="N",
        help="maximum seasonal AR order",
    )
    auto_arima_group.add_argument(
        "--max_Q",
        type=int,
        default=2,
        metavar="N",
        help="maximum seasonal MA order",
    )
    auto_arima_group.add_argument(
        "--max_D",
        type=int,
        default=1,
        metavar="N",
        help="maximum seasonal differencing order",
    )
    auto_arima_group.add_argument(
        "--alias",
        type=str,
        default="AutoARIMA",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_auto_mfles_args(parser):
    """Add Auto MFLES arguments."""
    auto_mfles_group = parser.add_argument_group("Auto MFLES", "Auto MFLES model-specific settings")

    # MFLES specific parameters
    auto_mfles_group.add_argument(
        "--season_length",
        type=int,
        nargs="*",
        default=24,
        metavar="N",
        help="Seasonal period(s). For hourly data: 24=daily, 168=weekly. If not specified, automatically determined",
    )
    auto_mfles_group.add_argument(
        "--n_windows",
        type=int,
        default=2,
        metavar="N",
        help="Number of windows for cross-validation",
    )
    auto_mfles_group.add_argument(
        "--step_size",
        type=int,
        default=None,
        metavar="N",
        help="Step size for rolling window validation. If None, equals test_size",
    )
    auto_mfles_group.add_argument(
        "--metric",
        type=str,
        default="smape",
        choices=["smape", "mase", "rmse", "mae", "mape"],
        metavar="METRIC",
        help="Metric for model selection",
    )
    auto_mfles_group.add_argument(
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Print detailed information during fitting",
    )
    auto_mfles_group.add_argument(
        "--alias",
        type=str,
        default="AutoMFLES",
        metavar="ALIAS",
        help="Model alias name",
    )
    auto_mfles_group.add_argument(
        "--prediction_intervals",
        type=str,
        default=None,
        metavar="INTERVALS",
        help="Prediction intervals configuration",
    )

    return parser


def _add_auto_tbats_args(parser):
    """Add Auto TBATS arguments."""
    auto_tbats_group = parser.add_argument_group("Auto TBATS", "Auto TBATS model-specific settings")

    # TBATS specific parameters
    auto_tbats_group.add_argument(
        "--season_length",
        type=int,
        nargs="+",
        default=[24],
        metavar="N",
        help="Seasonal period(s). For hourly data: 24=daily, 168=weekly. Can specify multiple (e.g., 24 168 for both daily and weekly patterns)",
    )
    auto_tbats_group.add_argument(
        "--use_boxcox",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use Box-Cox transformation. If None, automatically determined",
    )
    auto_tbats_group.add_argument(
        "--bc_lower_bound",
        type=float,
        default=0.0,
        metavar="BOUND",
        help="Lower bound for Box-Cox parameter",
    )
    auto_tbats_group.add_argument(
        "--bc_upper_bound",
        type=float,
        default=1.0,
        metavar="BOUND",
        help="Upper bound for Box-Cox parameter",
    )
    # TODO: default was None before, check if it works the new 'False' arg aswell
    auto_tbats_group.add_argument(
        "--use_trend",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use trend component. If None, automatically determined",
    )
    auto_tbats_group.add_argument(
        "--use_damped_trend",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use damped trend. If None, automatically determined",
    )
    auto_tbats_group.add_argument(
        "--use_arma_errors",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use ARMA errors",
    )
    auto_tbats_group.add_argument(
        "--alias",
        type=str,
        default="AutoTBATS",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_historic_average_args(parser):
    """Add Historic Average arguments."""
    historic_average_group = parser.add_argument_group(
        "Historic Average", "Historic Average model-specific settings"
    )

    historic_average_group.add_argument(
        "--alias",
        type=str,
        default="HistoricAverage",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_naive_args(parser):
    """Add Naive arguments."""
    naive_group = parser.add_argument_group("Naive", "Naive model-specific settings")

    naive_group.add_argument(
        "--alias", type=str, default="Naive", metavar="ALIAS", help="Model alias name"
    )

    return parser


def _add_seasonal_naive_args(parser):
    """Add Seasonal Naive arguments."""
    seasonal_naive_group = parser.add_argument_group(
        "Seasonal Naive", "Seasonal Naive model-specific settings"
    )

    seasonal_naive_group.add_argument(
        "--season_length",
        type=int,
        default=24,
        metavar="N",
        help="Seasonal period length (e.g., 24 for daily seasonality in hourly data)",
    )
    seasonal_naive_group.add_argument(
        "--alias",
        type=str,
        default="SeasonalNaive",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


def _add_seasonal_exponential_smoothing_args(parser):
    """Add Seasonal Exponential Smoothing arguments."""
    seasonal_exponential_smoothing_group = parser.add_argument_group(
        "Seasonal Exponential Smoothing",
        "Seasonal Exponential Smoothing model-specific settings",
    )

    seasonal_exponential_smoothing_group.add_argument(
        "--season_length",
        type=int,
        default=24,
        metavar="N",
        help="Seasonal period length (e.g., 24 for daily seasonality in hourly data)",
    )
    seasonal_exponential_smoothing_group.add_argument(
        "--prediction_intervals",
        type=str,
        default=None,
        metavar="INTERVALS",
        help="Prediction intervals configuration",
    )
    seasonal_exponential_smoothing_group.add_argument(
        "--alias",
        type=str,
        default="SeasonalExponentialSmoothingOptimized",
        metavar="ALIAS",
        help="Model alias name",
    )

    return parser


# Model configurations


def get_samformer_config():
    """
    Complete SAMFormer configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Deep learning config (optimizer, hyperparameters)

    - Loss landscape config (visualization and plotting)
    - SAMFormer model architecture

    - SAM optimization
    - GSAM optimization

    """
    # Start with loss landscape config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add SAMFormer-specific configurations
    parser = _add_samformer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_tsmixer_config():
    """
    Complete TSMixer configuration parser.

    Includes:


    - Base config (hardware, model, dataset, experiment)
    - Deep learning config (optimizer, hyperparameters)


    - Loss landscape config (visualization and plotting)
    - TSMixer model architecture


    - SAM optimization
    - GSAM optimization

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add TSMixer-specific configurations
    parser = _add_tsmixer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_tsmixer_ext_config():
    """
    Complete TSMixerExt configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - TSMixer base args (activation, num_blocks, dropout_rate, ff_dim, etc.)
    - TSMixerExt-specific args (extra_channels, hidden_channels, static_channels)
    - SAM optimization
    - GSAM optimization
    - Loss landscape config (visualization and plotting)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add TSMixer base args (shared with TSMixerExt)
    parser = _add_tsmixer_args(parser)

    # Add TSMixerExt-specific args
    parser = _add_tsmixer_ext_args(parser)

    # Override defaults that differ for TSMixerExt
    _override_argument_default(parser, "normalize_before", False)
    _override_argument_default(parser, "norm_type", "layer")

    # Add SAM/GSAM optimization
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_formers_config():
    """
    Complete Formers (Autoformer, Informer, Transformer) configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Common Transformer architecture (shared with PatchTST)
    - Formers-specific config (decoder, embeddings, etc.)
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add common Transformer architecture arguments
    parser = _add_formers_common_args(parser)

    # Add Formers-specific arguments
    parser = _add_formers_specific_args(parser)

    # Add SAM/GSAM optimization
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


# TODO: maybe just one get_config for all formers? If they all use the same args


def get_autoformer_config():
    """
    Complete Autoformer configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Common Transformer architecture (shared with other Formers)
    - Formers-specific config (decoder, embeddings, etc.)
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add common Transformer architecture arguments
    parser = _add_formers_common_args(parser)

    # Add Formers-specific arguments
    parser = _add_formers_specific_args(parser)

    # Add SAM/GSAM optimization
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_dlinear_config():
    """
    Complete DLinear configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - DLinear model architecture
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_dlinear_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_fedformer_config():
    """
    Complete FEDformer configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Common Transformer architecture (enc_in, d_model, n_heads, etc.)
    - Formers-specific config (decoder, embeddings, etc.)
    - FEDformer-specific config (version, modes, wavelets)
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_formers_common_args(parser)
    parser = _add_formers_specific_args(parser)
    parser = _add_fedformer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_informer_config():
    """
    Complete Informer configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Common Transformer architecture (shared with other Formers)
    - Formers-specific config (decoder, embeddings, distillation, etc.)
    - SAM/GSAM optimization
    - Loss landscape visualization

    Note: Informer uses the same arguments as other Formers (Autoformer, Transformer).
    The key Informer-specific feature is the `distil` parameter which controls
    whether to use distilling (ConvLayers) in the encoder for self-attention
    distillation, reducing complexity from O(L^2) to O(LlogL).
    """
    parser = get_base_config()

    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_formers_common_args(parser)
    parser = _add_formers_specific_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_itransformer_config():
    """
    Complete iTransformer configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Common Transformer architecture (enc_in, d_model, n_heads, etc.)
    - iTransformer-specific config (use_norm, class_strategy)
    - SAM/GSAM optimization
    - Loss landscape visualization

    Note: iTransformer is encoder-only, so it doesn't use decoder-specific
    parameters like dec_in, d_layers, label_len.
    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_formers_common_args(parser)
    parser = _add_itransformer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_linear_config():
    """
    Complete Linear configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Linear model architecture
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_linear_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_nlinear_config():
    """
    Complete NLinear (Normalization-Linear) configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - NLinear model architecture
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_nlinear_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_patchtst_config():
    """
    Complete PatchTST configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - Common Transformer architecture (shared with Formers)
    - PatchTST-specific config (patches, RevIN, etc.)
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add common Transformer architecture arguments
    parser = _add_formers_common_args(parser)

    # Add PatchTST-specific arguments
    parser = _add_patchtst_specific_args(parser)

    # Add SAM/GSAM optimization
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_moment_config():
    """
    Complete MOMENT configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - MOMENT model architecture (model name, freezing options)
    - SAM/GSAM optimization
    - Loss landscape visualization

    Note: MOMENT expects seq_len=512 (its context length).
    For shorter sequences, the engine handles padding automatically.
    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add MOMENT-specific configurations
    parser = _add_moment_args(parser)

    # Add SAM/GSAM optimization
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_timesfm_config():
    """
    Complete TimesFM configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - TimesFM model architecture (model name, forecast config options)
    - SAM/GSAM optimization (for potential fine-tuning)
    - Loss landscape visualization

    Note: TimesFM is primarily a zero-shot model. The deep learning and
    optimization configs are included for potential fine-tuning scenarios.
    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration (needed for potential fine-tuning)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add TimesFM-specific configurations
    parser = _add_timesfm_args(parser)

    # Add SAM/GSAM optimization (for potential fine-tuning)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # # Add loss landscape configuration
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_timekan_config():
    """
    Complete TimeKAN configuration parser.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - TimeKAN model architecture
    - SAM/GSAM optimization
    - Loss landscape visualization

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)
    parser = _add_timekan_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)
    # parser = _add_loss_landscape_args(parser)

    return parser


# def get_tritracknet_config():
#     """
#     Complete TriTrackNet configuration parser.
#
#     Includes:
#     - Base config (hardware, model, dataset, experiment)
#     - Time series forecast config (seq_len, pred_len)
#     - Deep learning config (optimizer, hyperparameters)
#     - TriTrackNet model architecture
#     - SAM optimization
#     - GSAM optimization
#     - Loss landscape config (visualization and plotting)
#
#     """
#     # Start with base config
#     parser = get_base_config()
#
#     # Add time series forecast config
#     parser = _add_time_series_forecast_args(parser)
#
#     # Add deep learning configuration
#     parser = _add_deep_learning_args(parser)
#
#     # Add TriTrackNet-specific configurations
#     parser = _add_tritracknet_args(parser)
#     parser = _add_sam_args(parser)
#     parser = _add_gsam_args(parser)
#
#     # Add loss landscape configuration
#     parser = _add_loss_landscape_args(parser)
#
#     return parser
#
#
# def get_timecf_config():
#     """
#     Complete TimeCF configuration parser.
#
#     Includes:
#     - Base config (hardware, model, dataset, experiment)
#     - Time series forecast config (seq_len, pred_len)
#     - Deep learning config (optimizer, hyperparameters)
#     - TimeCF model architecture
#     - SAMFre loss configuration
#     - SAM/GSAM optimization
#     - Loss landscape visualization
#
#     """
#     # Start with base config
#     parser = get_base_config()
#
#     # Add time series forecast config
#     parser = _add_time_series_forecast_args(parser)
#
#     # Add deep learning configuration
#     parser = _add_deep_learning_args(parser)
#
#     # Add TimeCF-specific configurations
#     parser = _add_timecf_args(parser)
#
#     # Add SAM/GSAM optimization
#     parser = _add_sam_args(parser)
#     parser = _add_gsam_args(parser)
#
#     # Add loss landscape configuration
#     parser = _add_loss_landscape_args(parser)
#
#     return parser


# def get_timesfm_config():
#     """TimesFM configuration parser."""
#     parser = get_base_config()
#     parser = _add_time_series_forecast_args(parser)
#     parser = _add_deep_learning_args(parser)
#     parser = _add_timesfm_args(parser)
#     return parser


# def get_patchtst_config():
#     """
#     Complete PatchTST configuration parser.
#
#     Includes:
#
#     - Base config (hardware, model, dataset, experiment)
#     - Deep learning config (optimizer, hyperparameters)
#
#     - Loss landscape config (visualization and plotting)
#     - PatchTST model architecture
#
#     - SAM optimization
#     - GSAM optimization
#
#     """
#     # Start with base config
#     parser = get_base_config()
#
#     # Add time series forecast config
#     parser = _add_time_series_forecast_args(parser)
#
#     # Add deep learning configuration
#     parser = _add_deep_learning_args(parser)
#
#     # Add PatchTST-specific configurations
#     parser = _add_patchtst_args(parser)
#     parser = _add_sam_args(parser)
#     parser = _add_gsam_args(parser)
#
#     # Add loss landscape configuration
#     parser = _add_loss_landscape_args(parser)
#
#     return parser


def get_auto_arima_config():
    """
    Complete Auto ARIMA configuration parser.

    Includes:


    - Base config (hardware, model, dataset, experiment)
    - StatsForecast config (frequency, parallel processing)

    - Auto ARIMA model-specific config (seasonal parameters, ARIMA orders)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto ARIMA-specific configurations
    parser = _add_auto_arima_args(parser)

    return parser


def get_automfles_config():
    """
    Complete Auto MFLES configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)

    - StatsForecast config (frequency, parallel processing)
    - Auto MFLES model-specific config (seasonal parameters, validation settings)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto MFLES-specific configurations
    parser = _add_auto_mfles_args(parser)

    return parser


def get_auto_tbats_config():
    """
    Complete Auto TBATS configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)

    - StatsForecast config (frequency, parallel processing)
    - Auto TBATS model-specific config (seasonal parameters, Box-Cox, trend, ARMA)

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto TBATS-specific configurations
    parser = _add_auto_tbats_args(parser)

    return parser


def get_historic_average_config():
    """
    Complete Historic Average configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)

    - StatsForecast config (frequency, parallel processing)
    - Historic Average model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_historic_average_args(parser)
    return parser


def get_naive_config():
    """
    Complete Naive configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)

    - StatsForecast config (frequency, parallel processing)
    - Naive model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_naive_args(parser)
    return parser


def get_seasonal_naive_config():
    """
    Complete Seasonal Naive configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)

    - StatsForecast config (frequency, parallel processing)
    - Seasonal Naive model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_seasonal_naive_args(parser)
    return parser


def get_seasonal_exponential_smoothing_config():
    """
    Complete Seasonal Exponential Smoothing configuration parser.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)

    - StatsForecast config (frequency, parallel processing)
    - Seasonal Exponential Smoothing model-specific config

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_seasonal_exponential_smoothing_args(parser)
    return parser


# TODO: TEST:


def _add_multi_split_experiment_args(parser):
    """Add multi split experiment arguments."""
    multi_split_group = parser.add_argument_group(
        "Multi Split Comparison", "Multi Split comparison settings"
    )

    multi_split_group.add_argument(
        "--num_splits",
        type=int,
        default=3,
        metavar="N",
        help="number of train/test splits for multi split experiment",
    )
    multi_split_group.add_argument(
        "--multi_split_experiment",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="enable multi split experiment mode",
    )

    return parser


def get_samformer_multi_split_experiment_config():
    """
    SAMFormer configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Deep learning config (optimizer, hyperparameters)
    - SAMFormer model architecture
    - SAM/GSAM optimization
    - Multi Split comparison settings

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add SAMFormer-specific configurations
    parser = _add_samformer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    # # Add loss landscape configuration (optional for analysis)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_patchtst_multi_split_experiment_config():
    """
    PatchTST configuration for multi split experiment.

    Includes:

    - Base config (hardware, model, dataset, experiment)
    - Deep learning config (optimizer, hyperparameters)

    - Common Transformer architecture (shared with Formers)
    - PatchTST-specific config (patches, etc.)

    - SAM/GSAM optimization
    - Multi Split comparison settings

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add common Transformer architecture arguments
    parser = _add_formers_common_args(parser)

    # Add PatchTST-specific arguments
    parser = _add_patchtst_specific_args(parser)

    # Add SAM/GSAM optimization
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    # # Add loss landscape configuration (optional for analysis)
    # parser = _add_loss_landscape_args(parser)

    return parser


def get_tsmixer_multi_split_experiment_config():
    """
    TSMixer configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - Deep learning config (optimizer, hyperparameters)
    - TSMixer model architecture
    - SAM/GSAM/FSAM optimization
    - Multi Split comparison settings

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add deep learning configuration
    parser = _add_deep_learning_args(parser)
    # Add scheduler configuration
    parser = _add_scheduler_args(parser)

    # Add TSMixer-specific configurations
    parser = _add_tsmixer_args(parser)
    parser = _add_sam_args(parser)
    parser = _add_gsam_args(parser)
    parser = _add_fsam_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_auto_arima_multi_split_experiment_config():
    """
    Auto ARIMA configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Auto ARIMA model-specific config (seasonal parameters, ARIMA orders)
    - Multi Split comparison settings

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto ARIMA-specific configurations
    parser = _add_auto_arima_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_automfles_multi_split_experiment_config():
    """
    Auto MFLES configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Auto MFLES model-specific config (seasonal parameters, validation settings)
    - Multi Split comparison settings

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto MFLES-specific configurations
    parser = _add_auto_mfles_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_auto_tbats_multi_split_experiment_config():
    """
    Auto TBATS configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Auto TBATS model-specific config (seasonal parameters, Box-Cox, trend, ARMA)
    - Multi Split comparison settings

    """
    # Start with base config
    parser = get_base_config()

    # Add time series forecast config
    parser = _add_time_series_forecast_args(parser)

    # Add StatsForecast configuration
    parser = _add_statsforecast_args(parser)

    # Add Auto TBATS-specific configurations
    parser = _add_auto_tbats_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_historic_average_multi_split_experiment_config():
    """
    Historic Average configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Historic Average model-specific config
    - Multi Split comparison settings

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_historic_average_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_naive_multi_split_experiment_config():
    """
    Naive configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Naive model-specific config
    - Multi Split comparison settings

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_naive_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_seasonal_naive_multi_split_experiment_config():
    """
    Seasonal Naive configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Seasonal Naive model-specific config
    - Multi Split comparison settings

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_seasonal_naive_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser


def get_seasonal_exponential_smoothing_multi_split_experiment_config():
    """
    Seasonal Exponential Smoothing configuration for multi split experiment.

    Includes:
    - Base config (hardware, model, dataset, experiment)
    - Time series forecast config (seq_len, pred_len)
    - StatsForecast config (frequency, parallel processing)
    - Seasonal Exponential Smoothing model-specific config
    - Multi Split comparison settings

    """
    parser = get_base_config()
    parser = _add_time_series_forecast_args(parser)
    parser = _add_statsforecast_args(parser)
    parser = _add_seasonal_exponential_smoothing_args(parser)

    # Add multi split experiment arguments
    parser = _add_multi_split_experiment_args(parser)

    return parser
