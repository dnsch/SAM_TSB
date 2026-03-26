from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple
import argparse

import numpy as np
import torch

from src.utils.paths import get_experiment_results_dir


class BaseExperiment(ABC):
    """
    Base class for all experiment types.

    Provides common infrastructure:
        - Configuration parsing
        - Logging setup
        - Seed management

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - get_results_subdir(): Return results subdirectory name
        - get_log_dir_components(args): Return log directory path components
        - setup_dataloader(args, logger): Setup and return dataloader
        - run(): Main entry point

    """

    def __init__(self):
        self.args: Optional[argparse.Namespace] = None
        self.log_dir: Optional[Path] = None
        self.logger = None

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def get_config_parser(self) -> argparse.ArgumentParser:
        """Return the argument parser for this experiment."""
        pass

    @abstractmethod
    def get_model_name(self):
        """Return the model name string."""
        pass

    @abstractmethod
    def get_results_subdir(self):
        """
        Return the subdirectory name under results/.

        Examples: 'single_split', 'multi_split'
        """
        pass

    @abstractmethod
    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """
        Return tuple of path components for log directory.

        Example: ('ModelName', 'dataset_name', 'seq_len_96_pred_len_24')
        """
        pass

    @abstractmethod
    def run(self):
        """Main entry point for the experiment."""
        pass

    # =========================================================================
    # Common Implementation
    # =========================================================================

    def get_config(self) -> Tuple[argparse.Namespace, Path, Any]:
        """Setup configuration, logging directory, and logger."""
        from src.utils.logging import get_logger

        parser = self.get_config_parser()
        args = parser.parse_args()
        args.model_name = self.get_model_name()
        args._parser = parser

        # Build log directory path
        base_dir = get_experiment_results_dir(self.get_results_subdir())
        components = self.get_log_dir_components(args)

        log_dir = base_dir.joinpath(*components)

        log_dir = log_dir / f"seed_{args.seed}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = get_logger(log_dir, __name__, f"record_s{args.seed}.log")
        logger.info(args)

        self.args = args
        self.log_dir = log_dir
        self.logger = logger

        return args, log_dir, logger

    def _setup_loss_function(self, args: argparse.Namespace):
        """Setup loss function from args if loss_name is specified."""
        if hasattr(args, "loss_name"):
            from src.utils.loss_functions import get_loss_function

            args.loss_fn = get_loss_function(args.loss_name)
        return args


def run_experiment(experiment_class: type):
    """
    Generic runner for any experiment class.
    Use in if __name__ == "__main__" blocks.
    """
    experiment = experiment_class()
    return experiment.run()
