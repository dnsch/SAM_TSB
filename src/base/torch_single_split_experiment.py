from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse

import torch
import numpy as np
import csv

torch.set_num_threads(3)

from src.base.base_experiment import BaseExperiment

from src.utils.setup import setup_seed, setup_dataloader, setup_optimizer_and_scheduler


class TorchSingleSplitExperiment(BaseExperiment):
    """
    Base class for PyTorch single split training experiments.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_model(args, dataloader): Create and return model
        - get_engine_class(): Return the engine class to use

    Optional overrides:
        - get_log_dir_suffix(args): Custom model name suffix
        - get_log_dir_params(args): Custom parameter string
        - get_dataloader_class(): Custom dataloader class
        - get_dataloader_kwargs(args): Custom dataloader kwargs
        - get_loss_function(): Custom loss function
        - get_metrics(): Custom metrics list
        - get_engine_kwargs(...): Custom engine kwargs
        - post_training_hooks(...): Custom post-training analysis

    """

    def __init__(self):
        super().__init__()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader_instance = None
        self.dataloader = None
        self.engine = None
        self.scaler = None
        self._input_channels = None

    # =========================================================================
    # Path Configuration (implements BaseExperiment abstract methods)
    # =========================================================================

    def get_results_subdir(self):
        """Results go under results/single_split/"""
        return "single_split"

    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """Return log directory path components. This function crafts the save
        path based on the SAM configs."""

        return (
            self.get_model_name(),  # model name for example "samformer"
            self.get_optimizer_variant(args),  # "SAM", "GSAM", "base", "FSAM"
            args.dataset,
            self.get_log_dir_params(args),
        )

    def get_optimizer_variant(self, args: argparse.Namespace) -> str:
        """Get the optimizer/training variant name for log directory."""
        if getattr(args, "sam", False):
            if getattr(args, "sam_adaptive", False):
                return "ASAM"  # Adaptive SAM
            return "SAM"
        elif getattr(args, "gsam", False):
            return "GSAM"
        elif getattr(args, "fsam", False):
            return "FSAM"
        return "base"

    # def get_log_dir_suffix(self, args: argparse.Namespace):
    #     """Get model name suffix based on optimizer type."""
    #     model_name = self.get_model_name()
    #     if getattr(args, "sam", False):
    #         return f"{model_name}SAM"
    #     elif getattr(args, "gsam", False):
    #         return f"{model_name}GSAM"
    #     return model_name

    def get_log_dir_params(self, args: argparse.Namespace):
        """Get parameter-specific part of log directory path."""
        base_params = f"seq_len_{args.seq_len}_pred_len_{args.pred_len}_bs_{args.batch_size}"
        if getattr(args, "sam", False):
            adaptive_str = "_adaptive" if getattr(args, "sam_adaptive", False) else ""
            return f"{base_params}_rho_{args.rho}{adaptive_str}"
        elif getattr(args, "gsam", False):
            return f"{base_params}_gsam_alpha_{args.gsam_alpha}_rho_max_{args.gsam_rho_max}"
        elif getattr(args, "fsam", False):
            return f"{base_params}_fsam_rho_{args.fsam_rho}_sigma_{args.fsam_sigma}_lmbda_{args.fsam_lmbda}"
        return base_params

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def create_model(self, args: argparse.Namespace, dataloader: Dict) -> torch.nn.Module:
        """Create and return the model instance."""
        pass

    @abstractmethod
    def get_engine_class(self):
        """Return the engine class to use for training."""
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    def get_dataloader_class(self):
        """Return the dataloader class to use."""
        from src.utils.dataloader import AutoformerDataloader

        return AutoformerDataloader

    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for dataloader initialization."""
        return {
            "dataset": args.dataset,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "seed": args.seed,
            "time_increment": args.time_increment,
            # "time_increment": getattr(args, "time_increment", 1),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "batch_size": args.batch_size,
            # add as sep arg, triggered by file path rather than an argument
            "multi_split": getattr(args, "multi_split", False),
            "shuffle_train_val": args.shuffle_train_val,
            "plot_dataset": args.plot_dataset,
        }

    def get_loss_function(self, args: argparse.Namespace):
        """Get the loss function from args or default to MSE."""
        if hasattr(args, "loss_name") and args.loss_name:
            from src.utils.loss_functions import get_loss_function

            return get_loss_function(args.loss_name)
        return torch.nn.MSELoss()

    def get_metrics(self):
        """Get metrics to track."""
        return ["mse", "mae", "mape", "rmse"]

    # TODO: delete
    # # TODO: change every argument to num_channels or let it be consistent in
    # # order to avoid this function
    # def get_revin_num_features(self, args: argparse.Namespace) -> Optional[int]:
    #     """
    #     Get the number of features for RevIN.
    #
    #     Override in subclasses to return the appropriate number of channels/features.
    #     Common options:
    #         - args.num_channels (SAMFormer, TSMixer)
    #         - args.enc_in (Transformers, PatchTST, DLinear)
    #
    #     If None is returned, the engine will try to get it from model.num_channels.
    #
    #     Args:
    #         args: Parsed arguments
    #
    #     Returns:
    #         Number of features for RevIN, or None to auto-detect from model
    #     """
    #     # Try common argument names in order of preference
    #     if hasattr(args, "num_channels"):
    #         return args.num_channels
    #     if hasattr(args, "enc_in"):
    #         return args.enc_in
    #     # Return None to let engine auto-detect from model
    #     return None

    def get_engine_kwargs(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        scaler,
        optimizer,
        scheduler,
        loss_fn: torch.nn.Module,
        log_dir: Path,
        logger,
    ) -> Dict[str, Any]:
        """Get kwargs for engine initialization."""
        return {
            "device": args.device,
            "model": model,
            "dataloader": dataloader,
            "scaler": scaler,
            "loss_fn": loss_fn,
            "lrate": args.lrate,
            "optimizer": optimizer,
            # RevIN parameters
            # "use_revin": getattr(args, "use_revin", False),
            "use_revin": args.use_revin,
            # "revin_affine": getattr(args, "revin_affine", False),
            "revin_affine": args.revin_affine,
            # "revin_subtract_last": getattr(args, "revin_subtract_last", False),
            "revin_subtract_last": args.revin_subtract_last,
            # Sharpness Aware Minimization
            # "sam": getattr(args, "sam", False),
            "sam": args.sam,
            # "gsam": getattr(args, "gsam", False),
            "gsam": args.gsam,
            "fsam": args.fsam,
            "scheduler": scheduler,
            "clip_grad_value": args.clip_grad_value,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "log_dir": log_dir,
            "logger": logger,
            "seed": args.seed,
            "metrics": self.get_metrics(),
        }

    def post_training_hooks(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        log_dir: Path,
        logger,
        loss_fn: torch.nn.Module,
    ):
        """Run after training/testing completes."""
        if getattr(args, "hessian_analysis", False):
            self._run_hessian_analysis(args, model, dataloader, log_dir, logger, loss_fn)

    # =========================================================================
    # Core Implementation
    # =========================================================================

    def create_engine(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        scaler,
        optimizer,
        scheduler,
        loss_fn: torch.nn.Module,
        log_dir: Path,
        logger,
    ):
        """Create and return the training engine."""
        EngineClass = self.get_engine_class()
        kwargs = self.get_engine_kwargs(
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
        )
        return EngineClass(**kwargs)

    def _run_hessian_analysis(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        log_dir: Path,
        logger,
        loss_fn: torch.nn.Module,
    ):
        """Run Hessian analysis if enabled."""
        from third_party.utils.pyhessian.pyhessian import hessian
        from third_party.utils.pyhessian.density_plot import get_esd_plot

        logger.info("Computing Hessian analysis...")

        use_cuda = torch.cuda.is_available() and "cuda" in args.device

        hessian_comp = hessian(model, loss_fn, dataloader=dataloader["train_loader"], cuda=use_cuda)

        trace_estimate = np.mean(hessian_comp.trace())
        num_params = sum(p.numel() for p in model.parameters())
        average_curvature = trace_estimate / num_params

        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1)
        max_eigenvalue = top_eigenvalues[0]

        print(f"Max eigenvalue: {max_eigenvalue}")
        print(f"Trace: {trace_estimate}")
        print(f"Parameters: {num_params}")
        print(f"Average curvature: {average_curvature}")

        save_dir = log_dir / "statistics"
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / f"hessian_metrics_s{args.seed}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["max_eigenvalue", "trace", "parameters", "average_curvature"])
            writer.writerow([max_eigenvalue, trace_estimate, num_params, average_curvature])

        logger.info(f"Hessian metrics saved to: {csv_path}")

        density_eigen, density_weight = hessian_comp.density()
        get_esd_plot(density_eigen, density_weight, log_dir)

    def run(self):
        """Runs the complete training pipeline."""

        # Setup configuration
        args, log_dir, logger = self.get_config()

        # Set seed for reproducibility
        setup_seed(args.seed)

        # Setup dataloader
        self.dataloader_instance, self.dataloader = setup_dataloader(
            dataloader_class=self.get_dataloader_class(),
            dataloader_kwargs=self.get_dataloader_kwargs(args),
        )
        self.scaler = self.dataloader_instance.get_scaler()

        # Create model
        self.model = self.create_model(args, self.dataloader)

        self.model.print_experiment_summary(args, logger)

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(
            self.model, args, self.dataloader, logger
        )

        # Get loss function and create engine
        loss_fn = self.get_loss_function(args)
        self.engine = self.create_engine(
            args,
            self.model,
            self.dataloader,
            self.scaler,
            self.optimizer,
            self.scheduler,
            loss_fn,
            log_dir,
            logger,
        )

        # Train or test
        if args.mode == "train":
            result = self.engine.train()
            logger.info(f"Training completed. Result: {result}")
        elif args.mode == "test":
            result = self.engine.evaluate(args.mode)
            logger.info(f"Testing completed. Result: {result}")

        # Post-training hooks
        self.post_training_hooks(args, self.model, self.dataloader, log_dir, logger, loss_fn)

        return result


def run_single_split_experiment(experiment_class: type):
    """Run a single split torch experiment."""
    from src.base.base_experiment import run_experiment

    return run_experiment(experiment_class)
