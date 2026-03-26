from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse
import numpy as np
import torch

torch.set_num_threads(3)

from src.base.base_experiment import BaseExperiment
from src.utils.setup import setup_seed, setup_dataloader, setup_optimizer_and_scheduler


class TorchMultiSplitExperiment(BaseExperiment):
    """
    Base class for PyTorch multi split experiment experiments.

    Runs the model on multiple train/test splits for fair comparison
    with statsforecast models that use rolling window evaluation.

    This evaluates the model across multiple randomly sampled train/test
    splits to ensure a robust comparison with statistical baselines.

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
        - get_num_splits(): Number of train/test splits to evaluate

    """

    def __init__(self):
        super().__init__()

        # Torch-specific state
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.dataloader_instance = None
        self.dataloaders_list: Optional[List[Dict]] = None
        self.scalers_list: Optional[List] = None
        self.engine = None
        self._input_channels = None

    # =========================================================================
    # Path Configuration (implements BaseExperiment abstract methods)
    # =========================================================================

    def get_results_subdir(self):
        """Results go under results/multi_split/"""
        return "multi_split"

    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """Return log directory path components. This function crafts the save
        path based on the SAM configs."""

        return (
            self.get_model_name(),  # model name for example "samformer"
            self.get_optimizer_variant(args),  # "SAM", "GSAM", "base"
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
        num_splits = self.get_num_splits(args)
        base_params = f"{base_params}_splits_{num_splits}"

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

    def get_num_splits(self, args: argparse.Namespace):
        """Get number of train/test splits for multi_split experiment."""
        return getattr(args, "num_splits", 3)

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
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "batch_size": args.batch_size,
            "multi_split": True,  # Enable multi_split experiment mode
            "num_splits": args.num_splits,  # Enable multi_split experiment mode
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
        split_idx=0,
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
            "use_revin": getattr(args, "use_revin", False),
            "revin_affine": getattr(args, "revin_affine", False),
            "revin_subtract_last": getattr(args, "revin_subtract_last", False),
            # Sharpness Aware Minimization
            "sam": args.sam,
            "gsam": args.gsam,
            "fsam": args.fsam,
            "scheduler": scheduler,
            "clip_grad_value": args.clip_grad_value,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "log_dir": log_dir / f"split_{split_idx}",
            "logger": logger,
            "seed": args.seed,
            "metrics": self.get_metrics(),
            # Disable plotting for individual splits to reduce clutter
            # "enable_plotting": getattr(args, "enable_plotting", False) and split_idx == 0,
            "enable_plotting": True,
        }

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
        split_idx=0,
    ):
        """Create and return the training engine."""
        EngineClass = self.get_engine_class()
        kwargs = self.get_engine_kwargs(
            args,
            model,
            dataloader,
            scaler,
            optimizer,
            scheduler,
            loss_fn,
            log_dir,
            logger,
            split_idx,
        )
        return EngineClass(**kwargs)

    def _aggregate_results(
        self, all_results: List[Dict[str, float]], logger
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results across all splits.

        Returns:
            Dictionary with 'mean', 'std', 'min', 'max' for each metric
        """
        if not all_results:
            return {}

        # Get all metric names
        metric_names = list(all_results[0].keys())

        aggregated = {}
        for metric in metric_names:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                aggregated[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "values": values,
                }

        # Log aggregated results
        logger.info("=" * 60)
        logger.info("AGGREGATED RESULTS ACROSS ALL SPLITS")
        logger.info("=" * 60)
        for metric, stats in aggregated.items():
            logger.info(
                f"{metric.upper()}: "
                f"Mean={stats['mean']:.6f} ± {stats['std']:.6f}, "
                f"Min={stats['min']:.6f}, Max={stats['max']:.6f}"
            )
        logger.info("=" * 60)

        return aggregated

    # TODO: reorganize

    def _log_split_shapes(self, split_dataloader: Dict, split_idx: int, logger):
        """Log the shapes and statistics of train, val, and test sets for a split."""
        logger.info(f"Dataset shapes and statistics for split {split_idx + 1}:")

        # Store data for combining train + val
        combined_data = {"x": None, "y": None}
        combined_info = {"num_samples": 0, "seq_len": 0, "pred_len": 0}

        # Store individual raw timesteps for correct combined calculation
        train_total_raw = None
        val_total_raw = None

        for name in ["train_loader", "val_loader", "test_loader"]:
            if name not in split_dataloader or split_dataloader[name] is None:
                logger.info(f"  {name}: None")
                continue

            dataset = split_dataloader[name].dataset
            x_data, y_data = None, None

            # Try to get data from dataset attributes
            if hasattr(dataset, "x") and hasattr(dataset, "y"):
                x_data = dataset.x
                y_data = dataset.y
            elif hasattr(dataset, "data_x") and hasattr(dataset, "data_y"):
                x_data = dataset.data_x
                y_data = dataset.data_y
            elif hasattr(dataset, "__len__") and len(dataset) > 0:
                # Fallback: stack samples from dataset
                try:
                    samples_x, samples_y = [], []
                    for i in range(len(dataset)):
                        sample = dataset[i]
                        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                            samples_x.append(sample[0])
                            samples_y.append(sample[1])
                    if samples_x and samples_y:
                        x_data = (
                            np.stack(samples_x)
                            if isinstance(samples_x[0], np.ndarray)
                            else torch.stack(samples_x)
                        )
                        y_data = (
                            np.stack(samples_y)
                            if isinstance(samples_y[0], np.ndarray)
                            else torch.stack(samples_y)
                        )
                except Exception as e:
                    logger.info(f"  {name}: Could not extract data - {e}")
                    continue

            if x_data is None or y_data is None:
                logger.info(f"  {name}: Could not access data")
                continue

            # Convert to numpy for statistics if needed
            if isinstance(x_data, torch.Tensor):
                x_data = x_data.numpy()
            if isinstance(y_data, torch.Tensor):
                y_data = y_data.numpy()

            # Calculate raw timesteps from sliding window samples
            num_samples = x_data.shape[0]
            seq_len = x_data.shape[-1]
            pred_len = y_data.shape[-1]

            x_raw_timesteps = num_samples + seq_len - 1
            y_raw_timesteps = num_samples + pred_len - 1
            total_raw_timesteps = num_samples + seq_len + pred_len - 1

            # Store individual raw timesteps for combined calculation
            if name == "train_loader":
                train_total_raw = total_raw_timesteps
            elif name == "val_loader":
                val_total_raw = total_raw_timesteps

            # Calculate statistics
            x_stats = {
                "min": np.min(x_data),
                "max": np.max(x_data),
                "mean": np.mean(x_data),
                "std": np.std(x_data),
            }
            y_stats = {
                "min": np.min(y_data),
                "max": np.max(y_data),
                "mean": np.mean(y_data),
                "std": np.std(y_data),
            }

            # Log shapes
            logger.info(f"  {name.upper()}:")
            logger.info(f"    Shape: X={tuple(x_data.shape)}, Y={tuple(y_data.shape)}")
            logger.info(
                f"    Raw timesteps: X={x_raw_timesteps}, Y={y_raw_timesteps}, "
                f"Total={total_raw_timesteps} (samples={num_samples}, seq_len={seq_len}, pred_len={pred_len})"
            )
            logger.info(
                f"    X stats: min={x_stats['min']:.6f}, max={x_stats['max']:.6f}, "
                f"mean={x_stats['mean']:.6f}, std={x_stats['std']:.6f}"
            )
            logger.info(
                f"    Y stats: min={y_stats['min']:.6f}, max={y_stats['max']:.6f}, "
                f"mean={y_stats['mean']:.6f}, std={y_stats['std']:.6f}"
            )

            # Accumulate train and val data for combined statistics
            if name in ["train_loader", "val_loader"]:
                if combined_data["x"] is None:
                    combined_data["x"] = x_data
                    combined_data["y"] = y_data
                    combined_info["seq_len"] = seq_len
                    combined_info["pred_len"] = pred_len
                else:
                    combined_data["x"] = np.concatenate([combined_data["x"], x_data], axis=0)
                    combined_data["y"] = np.concatenate([combined_data["y"], y_data], axis=0)
                combined_info["num_samples"] += num_samples

            # Log combined train + val after val_loader
            if name == "val_loader" and combined_data["x"] is not None:
                x_combined = combined_data["x"]
                y_combined = combined_data["y"]
                num_samples_combined = combined_info["num_samples"]
                seq_len_combined = combined_info["seq_len"]
                pred_len_combined = combined_info["pred_len"]

                # Calculate raw timesteps accounting for overlap
                # Train and val data OVERLAP by seq_len timesteps:
                #   train_data = data[:train_end]
                #   val_data = data[train_end - seq_len : val_end]
                # So combined unique timesteps = train_raw + val_raw - seq_len
                if train_total_raw is not None and val_total_raw is not None:
                    total_raw_combined = train_total_raw + val_total_raw - seq_len_combined
                    # X spans from 0 to val_end - pred_len, Y spans from seq_len to val_end
                    x_raw_combined = total_raw_combined - pred_len_combined
                    y_raw_combined = total_raw_combined - seq_len_combined
                else:
                    # Fallback if we couldn't get individual values
                    x_raw_combined = num_samples_combined + seq_len_combined - 1
                    y_raw_combined = num_samples_combined + pred_len_combined - 1
                    total_raw_combined = (
                        num_samples_combined + seq_len_combined + pred_len_combined - 1
                    )

                x_stats_combined = {
                    "min": np.min(x_combined),
                    "max": np.max(x_combined),
                    "mean": np.mean(x_combined),
                    "std": np.std(x_combined),
                }
                y_stats_combined = {
                    "min": np.min(y_combined),
                    "max": np.max(y_combined),
                    "mean": np.mean(y_combined),
                    "std": np.std(y_combined),
                }

                logger.info(f"  TRAIN + VAL (combined, accounting for {seq_len_combined} overlap):")
                logger.info(f"    Shape: X={tuple(x_combined.shape)}, Y={tuple(y_combined.shape)}")
                logger.info(
                    f"    Raw timesteps: X={x_raw_combined}, Y={y_raw_combined}, "
                    f"Total={total_raw_combined} (samples={num_samples_combined}, seq_len={seq_len_combined}, pred_len={pred_len_combined})"
                )
                logger.info(
                    f"    X stats: min={x_stats_combined['min']:.6f}, max={x_stats_combined['max']:.6f}, "
                    f"mean={x_stats_combined['mean']:.6f}, std={x_stats_combined['std']:.6f}"
                )
                logger.info(
                    f"    Y stats: min={y_stats_combined['min']:.6f}, max={y_stats_combined['max']:.6f}, "
                    f"mean={y_stats_combined['mean']:.6f}, std={y_stats_combined['std']:.6f}"
                )

    def run(self):
        """Run the complete multi split experiment pipeline."""

        # Setup configuration
        args, log_dir, logger = self.get_config()

        # Set seed for reproducibility
        setup_seed(args.seed)

        # Setup dataloader with multi_split=True
        self.dataloader_instance, self.dataloaders_list = setup_dataloader(
            dataloader_class=self.get_dataloader_class(),
            dataloader_kwargs=self.get_dataloader_kwargs(args),
        )

        # Get scalers list for each split
        self.scalers_list = self.dataloader_instance.get_scaler_list()

        # Get input channels
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        # Get loss function
        loss_fn = self.get_loss_function(args)

        # Log experiment info
        num_splits = len(self.dataloaders_list)
        logger.info("=" * 60)
        logger.info(f"MULTI SPLIT")
        logger.info(f"Model: {self.get_model_name().upper()}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"seq_len: {args.seq_len}, pred_len: {args.pred_len}")
        logger.info(f"Number of channels: {self._input_channels}")
        logger.info(f"Number of splits: {num_splits}")
        logger.info("=" * 60)

        # Store results from all splits
        all_results = []

        # Run training and evaluation for each split
        for split_idx, split_dataloader in enumerate(self.dataloaders_list):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"SPLIT {split_idx + 1}/{num_splits}")
            logger.info(f"{'=' * 60}")
            self._log_split_shapes(split_dataloader, split_idx, logger)

            import pdb

            # pdb.set_trace()

            # Get scaler for this split
            scaler = self.scalers_list[split_idx] if self.scalers_list else None

            # Create fresh model for each split
            self.model = self.create_model(args, split_dataloader)

            # Log model info only for first split
            if split_idx == 0:
                self.model.print_experiment_summary(args, logger)

            # Setup optimizer and scheduler for this split
            self.optimizer, self.scheduler = setup_optimizer_and_scheduler(
                self.model, args, split_dataloader, logger if split_idx == 0 else None
            )

            # Create split-specific log directory
            split_log_dir = log_dir / f"split_{split_idx}"
            split_log_dir.mkdir(parents=True, exist_ok=True)

            # Create engine for this split
            self.engine = self.create_engine(
                args,
                self.model,
                split_dataloader,
                scaler,
                self.optimizer,
                self.scheduler,
                loss_fn,
                log_dir,
                logger,
                split_idx,
            )

            # Train and evaluate
            if args.mode == "train":
                result = self.engine.train()
                logger.info(f"Split {split_idx + 1} completed. Result: {result}")
            elif args.mode == "test":
                result = self.engine.evaluate("test")
                logger.info(f"Split {split_idx + 1} test completed. Result: {result}")

            if result:
                all_results.append(result)

        # Aggregate and log final results
        aggregated_results = self._aggregate_results(all_results, logger)

        # Save aggregated results to file
        self._save_aggregated_results(aggregated_results, log_dir, logger)

        return aggregated_results

    def _save_aggregated_results(
        self, aggregated_results: Dict[str, Dict[str, float]], log_dir: Path, logger
    ):
        """Save aggregated results to CSV and JSON files."""
        import json
        import pandas as pd

        # Save as JSON
        json_path = log_dir / "aggregated_results.json"
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for metric, stats in aggregated_results.items():
            json_results[metric] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in stats.items()
                if k != "values"
            }
            json_results[metric]["values"] = [float(x) for x in stats.get("values", [])]

        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Aggregated results saved to {json_path}")

        # Save summary as CSV
        csv_path = log_dir / "aggregated_results_summary.csv"
        rows = []
        for metric, stats in aggregated_results.items():
            rows.append(
                {
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to {csv_path}")


def run_multi_split_experiment(experiment_class: type):
    """Run a multi split experiment experiment."""
    from src.base.base_experiment import run_experiment

    return run_experiment(experiment_class)
