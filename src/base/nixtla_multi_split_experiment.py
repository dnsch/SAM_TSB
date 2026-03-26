from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

from src.base.base_experiment import BaseExperiment
from src.utils.setup import setup_seed


class NixtlaMultiSplitComparison(BaseExperiment):
    """
    Base class for statsforecast-based multi split experiment experiments.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_statsforecast_model(args): Create and return the SF model

    Optional overrides:
        - get_sf_kwargs(args): Custom StatsForecast initialization kwargs
        - get_engine_kwargs(...): Custom NixtlaEngine kwargs

    """

    # =========================================================================
    # Path Configuration (implements BaseExperiment abstract methods)
    # =========================================================================

    def get_results_subdir(self):
        """Results go under results/multi_split/"""
        return "multi_split"

    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """Return log directory path components."""
        return (
            args.model_name,
            args.dataset,
            f"seq_len_{args.seq_len}_pred_len_{args.pred_len}",
        )

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
            "freq": args.freq,
            "n_jobs": getattr(args, "n_jobs", -1),
        }

    def get_engine_kwargs(
        self,
        args: argparse.Namespace,
        sf,
        data,
        logger,
        log_dir: Path,
    ) -> Dict[str, Any]:
        """Get kwargs for NixtlaEngine initialization."""
        from src.utils.loss_functions import get_loss_function

        return {
            "model": sf,
            "dataloader": data,
            "scaler": None,
            "pred_len": args.pred_len,
            "loss_fn": get_loss_function(args.loss_name),
            "backend": "statsforecast",
            "num_channels": data[0]["unique_id"].nunique(),
            "freq": getattr(args, "freq", "H"),
            "n_jobs": getattr(args, "n_jobs", -1),
            "logger": logger,
            "log_dir": log_dir,
            "seed": args.seed,
            "enable_plotting": getattr(args, "enable_plotting", True),
            "metrics": getattr(args, "metrics", None),
            "args": args,
        }

    # =========================================================================
    # Core Implementation
    # =========================================================================

    def setup_dataloader(self, args: argparse.Namespace, logger=None):
        """Setup statsforecast dataloader."""
        from src.utils.dataloader import StatsforecastDataloader

        dataloader_instance = StatsforecastDataloader(
            dataset=args.dataset,
            args=args,
            logger=logger,
            merge_train_val=True,
        )
        return dataloader_instance.get_dataloader()

    def run_experiments(
        self,
        data_list: List,
        args: argparse.Namespace,
        logger,
        log_dir: Path,
    ):
        """Execute training for each data entry."""
        from statsforecast import StatsForecast
        from src.base.nixtla_engine import NixtlaEngine

        results = []

        for idx, data in enumerate(data_list):
            self._log_experiment_start(idx, len(data_list))

            # Log dataset shapes and statistics
            self._log_split_shapes(data, idx, logger)
            import pdb

            # pdb.set_trace()

            # Create model and StatsForecast instance
            sf_model = self.create_statsforecast_model(args)
            sf = StatsForecast(models=[sf_model], **self.get_sf_kwargs(args))

            # Create experiment log dir
            experiment_log_dir = log_dir / f"experiment_{idx}"

            # Create and run engine
            engine_kwargs = self.get_engine_kwargs(args, sf, data, logger, experiment_log_dir)
            engine = NixtlaEngine(**engine_kwargs)
            result = engine.train()
            results.append(result)

            self._log_experiment_end(idx, len(data_list))

        return results

    def _log_experiment_start(self, idx: int, total: int):
        """Log experiment start."""
        print(f"\n{'=' * 60}")
        print(f"Processing Experiment {idx + 1}/{total}")
        print(f"{'=' * 60}\n")

    def _log_experiment_end(self, idx: int, total: int):
        """Log experiment end."""
        print(f"\nCompleted Experiment {idx + 1}/{total}\n")

    def _log_split_shapes(self, data: Tuple, split_idx: int, logger):
        """Log the shapes and statistics of train and test sets for a split."""
        import pandas as pd
        import numpy as np

        logger.info(f"Dataset shapes and statistics for split {split_idx + 1}:")

        # Statsforecast data is typically (train_df, test_df) or similar tuple/list
        dataset_names = ["train", "test"]

        # Handle different data structures
        if isinstance(data, (tuple, list)):
            datasets = list(data)
        elif isinstance(data, dict):
            dataset_names = list(data.keys())
            datasets = list(data.values())
        else:
            logger.info(f"  Unknown data format: {type(data)}")
            return

        for i, (name, df) in enumerate(zip(dataset_names, datasets)):
            if df is None:
                logger.info(f"  {name}: None")
                continue

            if not isinstance(df, pd.DataFrame):
                logger.info(f"  {name}: Not a DataFrame (type={type(df)})")
                continue

            # Get shape info
            n_rows = len(df)
            n_channels = df["unique_id"].nunique() if "unique_id" in df.columns else 1
            n_timesteps = n_rows // n_channels if n_channels > 0 else n_rows

            logger.info(f"  {name.upper()}:")
            logger.info(f"    Shape: timesteps={n_timesteps}, channels={n_channels}")

            # Calculate statistics for target column 'y'
            if "y" in df.columns:
                y_data = df["y"].values
                y_stats = {
                    "min": np.min(y_data),
                    "max": np.max(y_data),
                    "mean": np.mean(y_data),
                    "std": np.std(y_data),
                }
                logger.info(
                    f"    Y stats: min={y_stats['min']:.6f}, max={y_stats['max']:.6f}, "
                    f"mean={y_stats['mean']:.6f}, std={y_stats['std']:.6f}"
                )

            # Log date range if 'ds' column exists
            if "ds" in df.columns:
                logger.info(f"    Date range: {df['ds'].min()} to {df['ds'].max()}")

    def run(self):
        """Main entry point."""
        args, log_dir, logger = self.get_config()
        self._setup_loss_function(args)
        setup_seed(args.seed)

        data = self.setup_dataloader(args, logger)
        results = self.run_experiments(data, args, logger, log_dir)

        # Log results summary
        for idx, result in enumerate(results):
            logger.info(f"Experiment {idx}: {result}")

        return results


def run_multi_split_experiment(training_class: type):
    """Run a multi split experiment experiment."""
    from src.base.base_experiment import run_experiment

    return run_experiment(training_class)
