import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import Dict, Any, Tuple, Type, Optional
import argparse
import random


from src.optimizers.sam.sam import SAM
from src.optimizers.gsam.gsam import GSAM
from src.optimizers.fsam.fsam import FSAM
from src.optimizers.gsam.scheduler import LinearScheduler, ProportionScheduler

# =========================================================================
# Reproducibility
# =========================================================================


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# Setup Dataloader
# =========================================================================


def setup_dataloader(
    dataloader_class: Type,
    dataloader_kwargs: Dict[str, Any],
) -> Tuple[Any, Dict]:
    """Initialize and return dataloader instance and dataloader dict."""
    dataloader_instance = dataloader_class(**dataloader_kwargs)
    dataloader = dataloader_instance.get_dataloader()
    return dataloader_instance, dataloader


# =========================================================================
# Setup Scheduler
# =========================================================================


def _create_cosine_warm_restarts_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> CosineAnnealingWarmRestarts:
    """Create CosineAnnealingWarmRestarts scheduler."""
    return CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=getattr(args, "scheduler_T0", 5),
        T_mult=getattr(args, "scheduler_T_mult", 1),
        eta_min=getattr(args, "scheduler_eta_min", 1e-6),
        last_epoch=-1,
    )


def _create_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataloader: Dict,
) -> LinearScheduler:
    """Create LinearScheduler."""
    T_max = args.max_epochs * len(dataloader["train_loader"])
    return LinearScheduler(
        T_max=T_max,
        max_value=args.lrate,
        min_value=getattr(args, "scheduler_eta_min", 0.0),
        optimizer=optimizer,
    )


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    dataloader: Optional[Dict] = None,
):
    """
    Create scheduler based on args.scheduler.

    Args:
        optimizer: The optimizer
        args: Arguments (must have 'scheduler' attribute)
        dataloader: Required for 'linear' scheduler

    Returns:
        Scheduler instance
    """
    scheduler_type = getattr(args, "scheduler", "cosine_warm_restarts")

    if scheduler_type == "none":
        return None
    if scheduler_type == "linear":
        if dataloader is None:
            raise ValueError("LinearScheduler requires dataloader")
        return _create_linear_scheduler(optimizer, args, dataloader)
    else:
        # Default to cosine_warm_restarts
        return _create_cosine_warm_restarts_scheduler(optimizer, args)


# =========================================================================
# Setup Optimizer
# =========================================================================


def _load_base_optimizer(model, args, logger):
    """
    Loads the optimizer based on the choice provided in args.
    """
    try:
        optimizer_class = getattr(torch.optim, args.optimizer)

        if args.sam or getattr(args, "fsam", False):
            if logger:
                logger.info(f"Optimizer class: {optimizer_class}")
            return optimizer_class
        elif args.gsam:
            if logger:
                logger.info(f"Optimizer class: {optimizer_class}")
            optimizer = optimizer_class(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
            if logger:
                logger.info(optimizer)
            return optimizer
        else:
            # no Sharpness Aware Minimization
            optimizer = optimizer_class(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
            if logger:
                logger.info(optimizer)
            return optimizer
    except AttributeError:
        raise ValueError(f"Optimizer '{args.optimizer}' not found in torch.optim.")


def _setup_sam_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataloader: Dict,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Setup SAM optimizer with cosine annealing scheduler."""

    base_optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = SAM(
        params=model.parameters(),
        base_optimizer=base_optimizer_class,
        rho=args.rho,
        adaptive=args.sam_adaptive,
        lr=args.lrate,
        weight_decay=args.wdecay,
    )
    scheduler = _create_scheduler(optimizer, args, dataloader)
    return optimizer, scheduler


def _setup_gsam_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataloader: Dict,
    base_optimizer: torch.optim.Optimizer,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Setup GSAM optimizer with linear LR scheduler and proportion rho scheduler."""

    # Note: GSAM uses its own LinearScheduler + ProportionScheduler combination
    # scheduler arg ignored here

    T_max = args.max_epochs * len(dataloader["train_loader"])

    lr_scheduler = LinearScheduler(
        T_max=T_max,
        max_value=args.lrate,
        min_value=0.0,
        optimizer=base_optimizer,
    )

    rho_scheduler = ProportionScheduler(
        pytorch_lr_scheduler=lr_scheduler,
        max_lr=args.lrate,
        min_lr=0.0,
        max_value=args.gsam_rho_max,
        min_value=args.gsam_rho_min,
    )

    optimizer = GSAM(
        params=model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        gsam_alpha=args.gsam_alpha,
        rho_scheduler=rho_scheduler,
        adaptive=args.gsam_adaptive,
    )
    return optimizer, lr_scheduler


def _setup_fsam_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataloader: Dict,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Setup FSAM optimizer with cosine annealing scheduler."""

    base_optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = FSAM(
        params=model.parameters(),
        base_optimizer=base_optimizer_class,
        rho=args.fsam_rho,
        sigma=args.fsam_sigma,
        lmbda=args.fsam_lmbda,
        adaptive=getattr(args, "fsam_adaptive", False),
        lr=args.lrate,
        weight_decay=args.wdecay,
    )
    scheduler = _create_scheduler(optimizer, args, dataloader)
    return optimizer, scheduler


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataloader: Dict,
    logger,
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Setup optimizer and learning rate scheduler with SAM/GSAM/FSAM support.
    """
    if getattr(args, "sam", False):
        return _setup_sam_optimizer(model, args, dataloader)
    elif getattr(args, "fsam", False):
        return _setup_fsam_optimizer(model, args, dataloader)
    elif getattr(args, "gsam", False):
        base_optimizer = _load_base_optimizer(model, args, logger)
        return _setup_gsam_optimizer(model, args, dataloader, base_optimizer)
    else:
        optimizer = _load_base_optimizer(model, args, logger)
        scheduler = _create_scheduler(optimizer, args, dataloader)
        return optimizer, scheduler
