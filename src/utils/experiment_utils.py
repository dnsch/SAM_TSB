from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[2]))
# from src.engines.samformer_engine import SAMFormer_Engine

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

# ============================================================

# Experiments

# ============================================================


def get_engine(
    model_name,
    device,
    model,
    dataloader,
    scaler,
    loss_fn,
    optimizer,
    lr_scheduler,
    log_dir,
    logger,
    args,
):
    """
    Function that creates and returns the appropriate engine based on model_name.

    Args:
        model_name: Name of the model (e.g., 'samformer')
        device: Device to run on
        model: The model instance
        dataloader: Dictionary containing train/val/test dataloaders
        scaler: Scaler for data normalization
        loss_fn: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        log_dir: Directory for logging
        logger: Logger object
        args: Arguments object containing hyperparameters

    Returns:
        engine: The appropriate engine instance
    """
    # TODO: use try to get num_channels and issue warning if that's unsuccessful
    if model_name == "samformer":
        engine = SAMFormer_Engine(
            device=device,
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            loss_fn=loss_fn,
            lrate=args.lrate,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            clip_grad_value=args.clip_grad_value,
            max_epochs=args.max_epochs,
            patience=args.patience,
            log_dir=log_dir,
            logger=logger,
            seed=args.seed,
            batch_size=args.batch_size,
            num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
            pred_len=args.horizon,
            no_sam=args.no_sam,
            use_revin=args.use_revin,
            gsam=args.gsam,
            plot_attention=args.plot_attention,
        )
    # elif model_name.lower() == "transformer":
    #     engine = Transformer_Engine(...)
    # elif model_name.lower() == "lstm":
    #     engine = LSTM_Engine(...)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return engine


def run_experiments_on_dataloader_list(
    dataloader_instance,
    dataloader_list,
    args,
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    log_dir,
    logger,
):
    """
    Execute training/evaluation for each dataloader in dataloader_list.

    Args:
        dataloader_list: List of dictionaries containing train/val/test dataloaders
        args: Arguments object containing hyperparameters and configuration
        model: The model to train/evaluate
        loss_fn: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        log_dir: Directory for logging
        logger: Logger object

    Returns:
        results: List of results from each experiment
    """
    # Get the scaler list
    scaler_list = dataloader_instance.get_scaler_list()

    results = []

    # Iterate through each dataloader
    for idx, dataloader in enumerate(dataloader_list):
        print(f"\n{'=' * 60}")
        print(f"Processing Experiment {idx + 1}/{len(dataloader_list)}")
        print(f"{'=' * 60}\n")

        # Get the corresponding scaler for this dataloader
        scaler = scaler_list[idx] if idx < len(scaler_list) else None

        # Create experiment-specific log directory
        experiment_log_dir = log_dir / f"experiment_{idx}"

        # TODO: put this in a separate function
        # Create the engine
        engine = get_engine(
            model_name=args.model_name,
            device=args.device,
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_dir=experiment_log_dir,
            logger=logger,
            args=args,
        )

        # Run train or test based on mode
        if args.mode == "train":
            result = engine.train()
        elif args.mode == "test":
            result = engine.evaluate(args.mode)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        results.append(result)

        print(f"\nCompleted Experiment {idx + 1}/{len(dataloader_list)}\n")

    return results


def compute_mean_forecasts(
    preds: torch.Tensor, labels: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean (and std) forecasts over overlapping rolling windows.
    Reconstruct ground truth as a single continuous time series.

    For rolling window forecasts:
    - Window i predicts timesteps i to i + pred_len - 1
    - Each global timestep t has multiple predictions from different windows
    - Ground truth at each timestep is identical across windows (no averaging needed)

    Args:
        preds: shape [num_windows, num_variables, pred_len]
        labels: shape [num_windows, num_variables, pred_len]

    Returns:
        mean_preds: shape [total_length, num_variables]
        std_preds: shape [total_length, num_variables]
        ground_truth: shape [total_length, num_variables]
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    num_windows, num_variables, pred_len = preds.shape
    total_length = num_windows + pred_len - 1

    # Collect all forecasts per timestep
    preds_per_timestep = [[] for _ in range(total_length)]

    for window_idx in range(num_windows):
        for pos in range(pred_len):
            global_t = window_idx + pos
            preds_per_timestep[global_t].append(preds[window_idx, :, pos])

    # Compute prediction statistics
    mean_preds = np.zeros((total_length, num_variables))
    std_preds = np.zeros((total_length, num_variables))

    for t in range(total_length):
        preds_stack = np.stack(preds_per_timestep[t], axis=0)  # [count, num_vars]
        mean_preds[t] = preds_stack.mean(axis=0)
        std_preds[t] = preds_stack.std(axis=0)

    # Reconstruct ground truth (labels are identical at each timestep, just need to stitch together)
    # First pred_len-1 timesteps: take from window 0, positions 0 to pred_len-2
    # Remaining timesteps: take from each window's last position (or any position that covers it)
    ground_truth = np.zeros((total_length, num_variables))

    # Fill from window 0 for the first pred_len timesteps
    ground_truth[:pred_len] = labels[0, :, :].T  # [pred_len, num_vars]

    # Fill remaining timesteps from subsequent windows (taking the last position of each)
    for window_idx in range(1, num_windows):
        global_t = window_idx + pred_len - 1
        ground_truth[global_t] = labels[window_idx, :, -1]

    return mean_preds, std_preds, ground_truth
