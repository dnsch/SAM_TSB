from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse

import torch

from src.base.base_experiment import BaseExperiment
from src.utils.setup import setup_seed, setup_dataloader


class NixtlaSingleSplitExperiment(BaseExperiment):
    """
    Base class for statsforecast-based single split experiments.

    Runs statsforecast models (Naive, SeasonalNaive, ARIMA, etc.)
    with single split window evaluation on the test set, matching the
    evaluation strategy of TorchSingleSplitExperiment.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_statsforecast_model(args): Create and return the SF model

    Optional overrides:
        - get_sf_kwargs(args): Custom StatsForecast initialization kwargs
        - get_log_dir_params(args): Custom parameter string for log directory
        - get_metrics(): Custom metrics list
        - get_engine_kwargs(...): Custom engine kwargs
        - get_dataloader_class(): Custom dataloader class
        - get_dataloader_kwargs(args): Custom dataloader kwargs

    """

    def __init__(self):
        super().__init__()
        self.dataloader_instance = None
        self.dataloader = None
        self.scaler = None
        self.engine = None
        self._input_channels = None

    # =========================================================================
    # Path Configuration (implements BaseExperiment abstract methods)
    # =========================================================================

    def get_results_subdir(self):
        """Results go under results/single_split/"""
        return "single_split"

    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """Return log directory path components."""
        return (
            self.get_model_name(),
            args.dataset,
            self.get_log_dir_params(args),
        )

    def get_log_dir_params(self, args: argparse.Namespace):
        """Get parameter-specific part of log directory path."""
        return f"seq_len_{args.seq_len}_pred_len_{args.pred_len}"

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def create_statsforecast_model(self, args: argparse.Namespace):
        """Create and return the statsforecast model instance."""
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    def get_sf_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for StatsForecast initialization."""
        return {
            "freq": getattr(args, "freq", "h"),
            "n_jobs": getattr(args, "n_jobs", 1),
        }

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
            "time_increment": getattr(args, "time_increment", 1),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "batch_size": 1,
            "multi_split": False,
        }

    def get_metrics(self) -> List[str]:
        """Get metrics to track."""
        return ["mse", "mae", "rmse", "mape"]

    def get_loss_function(self) -> torch.nn.Module:
        """Get the loss function."""
        return torch.nn.MSELoss()

    def get_single_split_stride(self, args: argparse.Namespace):
        """Get stride for single split window evaluation."""
        return getattr(args, "single_split_stride", args.pred_len)

    def get_engine_class(self):
        """Return the engine class to use."""
        from src.base.nixtla_engine import NixtlaEngine

        return NixtlaEngine

    def get_engine_kwargs(
        self,
        args: argparse.Namespace,
        sf,
        dataloader: Dict,
        dataloader_instance,
        scaler,
        loss_fn: torch.nn.Module,
        log_dir: Path,
        logger,
    ) -> Dict[str, Any]:
        """Get kwargs for NixtlaEngine initialization."""
        return {
            "model": sf,
            "dataloader": dataloader,
            "dataloader_instance": dataloader_instance,
            "scaler": scaler,
            "loss_fn": loss_fn,
            "backend": "statsforecast",
            "num_channels": self._input_channels,
            "pred_len": args.pred_len,
            "seq_len": args.seq_len,
            "freq": getattr(args, "freq", "h"),
            "n_jobs": getattr(args, "n_jobs", 1),
            "log_dir": log_dir,
            "logger": logger,
            "seed": args.seed,
            "enable_plotting": getattr(args, "enable_plotting", True),
            "metrics": self.get_metrics(),
            "evaluation_mode": "single_split",
            "single_split_stride": self.get_single_split_stride(args),
            "model_factory": lambda: self.create_statsforecast_model(args),
            "sf_kwargs": self.get_sf_kwargs(args),
            "alias": getattr(args, "alias", None),
            "args": args,
        }

    # =========================================================================
    # Core Implementation
    # =========================================================================

    def create_statsforecast(self, args: argparse.Namespace):
        """Create StatsForecast instance with the model."""
        from statsforecast import StatsForecast

        sf_model = self.create_statsforecast_model(args)
        sf = StatsForecast(models=[sf_model], **self.get_sf_kwargs(args))
        return sf

    def create_engine(
        self,
        args: argparse.Namespace,
        sf,
        dataloader: Dict,
        dataloader_instance,
        scaler,
        loss_fn: torch.nn.Module,
        log_dir: Path,
        logger,
    ):
        """Create and return the engine."""
        EngineClass = self.get_engine_class()
        kwargs = self.get_engine_kwargs(
            args, sf, dataloader, dataloader_instance, scaler, loss_fn, log_dir, logger
        )
        return EngineClass(**kwargs)

    def run(self):
        """Main entry point - runs the complete evaluation pipeline."""
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

        # Get input channels from dataloader
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        # Log experiment info
        logger.info("=" * 60)
        logger.info(f"Model: {self.get_model_name().upper()}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"seq_len: {args.seq_len}, pred_len: {args.pred_len}")
        logger.info(f"Number of channels: {self._input_channels}")
        logger.info("=" * 60)

        # Create StatsForecast instance
        sf = self.create_statsforecast(args)

        # Get loss function and create engine
        loss_fn = self.get_loss_function()
        self.engine = self.create_engine(
            args,
            sf,
            self.dataloader,
            self.dataloader_instance,
            self.scaler,
            loss_fn,
            log_dir,
            logger,
        )

        # Run training (which performs single split window evaluation)
        result = self.engine.train()

        # Log final results
        logger.info("=" * 60)
        logger.info("Final Test Results:")
        for metric_name, value in result.items():
            logger.info(f"{metric_name.upper()}: {value:.6f}")
        logger.info("=" * 60)

        return result


def run_nixtla_single_split_experiment(experiment_class: type):
    """Run a nixtla single_split experiment."""
    from src.base.base_experiment import run_experiment

    return run_experiment(experiment_class)
