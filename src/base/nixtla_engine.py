from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from src.utils.metrics import (
    get_metric_objects,
    get_metric_name_from_object,
)
from src.utils.model_utils import statsforecast_to_tensor
from src.utils.experiment_utils import compute_mean_forecasts
from src.utils.plotting import plot_mean_forecasts


# TODO: make sure the methods in the single split experiment get all the date as
# input, just as the other models. This includes train+val and the test input
# (which I think is the last input sequence from val anyways if we don't shuffle
#     the datasets)
class NixtlaEngine:
    """
    Engine for Nixtla-based models (statsforecast, neuralforecast, mlforecast).

    Supports two evaluation modes:
        - "multi_split": Fit once on train data, predict test (for multi_split experiment)
        - "single_split": Single split window evaluation (for single split experiments)

    """

    def __init__(
        self,
        model,
        dataloader,
        dataloader_instance=None,
        scaler=None,
        loss_fn=None,
        backend="statsforecast",
        num_channels=1,
        pred_len=1,
        seq_len=96,
        freq=None,
        n_jobs=-1,
        log_dir=None,
        logger=None,
        seed=1,
        enable_plotting=True,
        metrics=None,
        evaluation_mode="multi_split",
        single_split_stride=None,
        model_factory=None,
        sf_kwargs=None,
        alias=None,
        args=None,
        **kwargs,
    ):
        # Core components
        self.model = model
        self._dataloader = dataloader
        self._dataloader_instance = dataloader_instance
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._backend = backend
        self._num_channels = num_channels
        self._pred_len = pred_len
        self._seq_len = seq_len
        self._freq = freq
        self._n_jobs = n_jobs
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._enable_plotting = enable_plotting
        self._alias = alias or (args.alias if args else None)

        # Evaluation mode
        self._evaluation_mode = evaluation_mode
        # TODO: add this as argument
        # self._single_split_stride = single_split_stride or pred_len
        self._single_split_stride = 1
        self._model_factory = model_factory
        self._sf_kwargs = sf_kwargs or {"freq": freq, "n_jobs": n_jobs}

        # Initialize metrics
        self._metrics = self._initialize_metrics(metrics)

        if self._save_path:
            self._plot_path = self._save_path / "plots"
            self._plot_path.mkdir(parents=True, exist_ok=True)

            # Add statistics path
            self._stats_path = self._save_path / "statistics"
            self._stats_path.mkdir(parents=True, exist_ok=True)
        else:
            self._plot_path = None
            self._stats_path = None

        # Initialize tracking variables

        self._total_train_time = 0.0

        # Initialize backend-specific components
        self._initialize_backend(backend, freq, n_jobs)

        # Log initialization info
        if self._logger:
            self._logger.info(f"Backend: {self._backend}")
            self._logger.info(f"Evaluation mode: {self._evaluation_mode}")
            if self._loss_fn:
                self._logger.info(f"Loss function: {self._loss_fn.__class__.__name__}")
            self._logger.info(f"Prediction length: {self._pred_len}")
            self._logger.info(f"Number of channels: {self._num_channels}")

    # ==========================================================================
    # Backend Initialization
    # ==========================================================================

    def _initialize_backend(self, backend: str, freq: Optional[str], n_jobs: int):
        """Initialize backend-specific components."""
        if backend == "statsforecast":
            self._obj = self.model
            self._fit_fn = lambda df, X: self.model.fit(df=df)
            self._forecast_fn = lambda h, X, level=None: self.model.predict(h=h)

        elif backend == "neuralforecast":
            from neuralforecast import NeuralForecast

            models = self.model if isinstance(self.model, (list, tuple)) else [self.model]
            self._obj = NeuralForecast(models=list(models), freq=freq)
            self._fit_fn = lambda df, X: self._obj.fit(df=df, val_size=0)
            self._forecast_fn = lambda h, X, level=None: self._obj.predict(h=h, futr_df=X)

        elif backend == "mlforecast":
            from mlforecast import MLForecast

            self._obj = (
                self.model
                if isinstance(self.model, MLForecast)
                else MLForecast(models=self.model, freq=freq)
            )
            self._fit_fn = lambda df, X: self._obj.fit(
                df=df, id_col="unique_id", time_col="ds", target_col="y"
            )
            self._forecast_fn = lambda h, X, level=None: self._obj.predict(h=h, new_df=X)

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def _to_numpy(self, tensors):
        """Convert tensor(s) to numpy array(s)."""
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        if isinstance(tensors, torch.Tensor):
            return tensors.detach().cpu().numpy()
        return tensors

    def _to_tensor(self, data):
        """Convert data to tensor(s)."""
        if isinstance(data, list):
            return [torch.tensor(item, dtype=torch.float32) for item in data]
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32)
        return data

    def _inverse_transform(self, tensors):
        """Apply inverse transformation using the scaler."""
        if self._scaler is None:
            return tensors

        inv = lambda tensor: self._scaler.inverse_transform(tensor)
        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        return inv(tensors)

    # ==========================================================================
    # Data Conversion
    # ==========================================================================

    def _wide_to_long(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        """Convert wide format DataFrame to statsforecast long format."""
        df_reset = df_wide.reset_index()
        date_col = df_reset.columns[0]
        df_long = df_reset.melt(id_vars=[date_col], var_name="unique_id", value_name="y")
        df_long = df_long.rename(columns={date_col: "ds"})
        df_long = df_long[["unique_id", "ds", "y"]]
        return df_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def _forecast_to_tensor(
        self,
        forecast: pd.DataFrame,
        pred_len: int,
        num_channels: int,
    ) -> torch.Tensor:
        """
        Convert statsforecast prediction to tensor format.

        Args:
            forecast: DataFrame with columns [unique_id, ds, <model_name>]
            pred_len: Prediction length
            num_channels: Number of channels/variables

        Returns:
            Tensor of shape [1, channels, pred_len]
        """
        model_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
        forecast_wide = forecast.pivot(index="ds", columns="unique_id", values=model_col)
        values = forecast_wide.values.T
        return torch.FloatTensor(values).unsqueeze(0)

    def _get_historical_data_long(self, end_idx: int) -> pd.DataFrame:
        """
        Get historical data up to end_idx in statsforecast long format.
        Applies scaling if scaler exists.
        """
        if self._dataloader_instance is None:
            raise ValueError("dataloader_instance required for single split window evaluation")

        df_raw = self._dataloader_instance.df_raw
        hist_df = df_raw.iloc[:end_idx].copy()

        if self._scaler is not None:
            hist_scaled = self._scaler.transform(hist_df.values)
            hist_df = pd.DataFrame(hist_scaled, index=hist_df.index, columns=hist_df.columns)

        return self._wide_to_long(hist_df)

    # ==========================================================================
    # Model Saving and Loading
    # ==========================================================================

    def save_model(self, save_path: Path):
        """Save final model."""
        save_path.mkdir(parents=True, exist_ok=True)
        saved_models_path = save_path / "saved_models"
        saved_models_path.mkdir(parents=True, exist_ok=True)

        filename = saved_models_path / f"final_model_s{self._seed}.pkl"

        if self._backend == "statsforecast":
            self.model.save(path=filename, max_size=None, trim=None)
        else:
            import pickle

            with open(filename, "wb") as f:
                pickle.dump(self._obj, f)

        self._logger.info(f"Model saved to {filename}")

    def load_model(self, save_path: Path):
        """Load saved model."""
        saved_models_path = save_path / "saved_models"
        filename = saved_models_path / f"final_model_s{self._seed}.pkl"

        if self._backend == "statsforecast":
            self.model.load(filename)
        else:
            import pickle

            with open(filename, "rb") as f:
                self._obj = pickle.load(f)

        self._logger.info(f"Model loaded from {filename}")

    # ==========================================================================
    # Data Preparation
    # ==========================================================================

    def _get_train_data(self):
        """Get training data."""
        if isinstance(self._dataloader, dict):
            return self._dataloader.get("train_loader") or self._dataloader[0]
        elif isinstance(self._dataloader, (list, tuple)):
            return self._dataloader[0]
        return self._dataloader

    def _get_data(self, mode: str):
        """Get data by mode."""
        if isinstance(self._dataloader, dict):
            key = f"{mode}_loader"
            return self._dataloader.get(key)
        elif isinstance(self._dataloader, (list, tuple)):
            if mode == "train":
                return self._dataloader[0]
            elif mode in ["val", "test"]:
                return self._dataloader[1] if len(self._dataloader) > 1 else None
        return None

    def _prepare_test_data(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare test data for evaluation.
        Reshapes data to format: (batch, channels, pred_len) to match TorchEngine.
        """
        if preds.dim() == 2:
            preds = preds.reshape(preds.shape[0], self._num_channels, self._pred_len)
        if labels.dim() == 2:
            labels = labels.reshape(labels.shape[0], self._num_channels, self._pred_len)
        return preds, labels

    # ==========================================================================
    # Metrics
    # ==========================================================================

    def _initialize_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Initialize metric calculators from metric name strings."""
        metric_names = metric_names if metric_names is not None else []

        if not metric_names:
            return {}

        try:
            metric_objects = get_metric_objects(metric_names)
            metric_calculators = {}

            for metric_obj in metric_objects:
                metric_name = get_metric_name_from_object(metric_obj)
                metric_calculators[metric_name] = metric_obj

            return metric_calculators

        except ValueError as e:
            if self._logger:
                self._logger.error(str(e))
            raise
        except ImportError as e:
            if self._logger:
                self._logger.warning(str(e))
                self._logger.warning("Additional metrics will not be available.")
            return {}

    def _get_metric_names(self) -> List[str]:
        """Return list of metric names to compute."""
        metric_names = [self._get_loss_name()]
        metric_names.extend(self._metrics.keys())
        return list(set(metric_names))

    def _get_loss_name(self):
        """Return the name of the loss metric."""
        if self._loss_fn is None:
            return "mse"

        loss_class_name = self._loss_fn.__class__.__name__.lower()

        if "mse" in loss_class_name:
            return "mse"
        elif "mae" in loss_class_name or "l1" in loss_class_name:
            return "mae"
        elif "huber" in loss_class_name:
            return "huber"
        return "loss"

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        if self._loss_fn is None:
            return torch.nn.MSELoss()(pred, target)
        return self._loss_fn(pred, target)

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics: loss + additional metrics."""
        metrics = {}
        loss_name = self._get_loss_name()

        # Always compute the loss
        loss = self._compute_loss(pred, target)
        metrics[loss_name] = loss.item() if isinstance(loss, torch.Tensor) else loss

        # Compute additional metrics
        for metric_name, metric_object in self._metrics.items():
            try:
                metric_value = metric_object(pred, target)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                metrics[metric_name] = metric_value
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Failed to compute metric '{metric_name}': {e}")
                metrics[metric_name] = float("nan")

        return metrics

    def _compute_test_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute test metrics and log detailed results."""
        preds, labels = self._prepare_test_data(preds, labels)
        metrics = self._compute_metrics(preds, labels)

        # Log all metrics
        for name, value in metrics.items():
            self._logger.info(f"Test {name.upper()}: {value:.4f}")

        # Per-horizon metrics
        avg_metrics = self._compute_per_horizon_metrics(preds, labels)

        # Save evaluation CSV
        self._save_evaluation_csv(avg_metrics, self._total_train_time)

        if self._enable_plotting:
            # New plot: mean forecasts with confidence bands
            mean_preds, std_preds, ground_truth = compute_mean_forecasts(preds, labels)

            plot_mean_forecasts(
                mean_preds,
                std_preds,
                ground_truth,
                plot_path=self._plot_path,
                pred_len=self._pred_len,
                plot_all_variables=False,
            )

            plot_mean_forecasts(
                mean_preds,
                std_preds,
                ground_truth,
                plot_path=self._plot_path,
                show_percent=100.0,
                plot_all_variables=False,
            )

        return metrics

    def _compute_per_horizon_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute and log per-horizon metrics."""
        pred_len = preds.shape[2]
        all_metric_names = self._get_metric_names()
        horizon_metrics = {metric: [] for metric in all_metric_names}

        for i in range(pred_len):
            horizon_pred = preds[:, :, i].contiguous()
            horizon_true = labels[:, :, i].contiguous()

            horizon_result = self._compute_metrics(horizon_pred, horizon_true)

            metric_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in horizon_result.items()])
            self._logger.info(f"Horizon {i + 1}, {metric_str}")

            for metric_name, value in horizon_result.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                horizon_metrics[metric_name].append(value)

        # Log average across horizons
        avg_metrics = {f"avg_{k}": np.mean(v) for k, v in horizon_metrics.items()}
        avg_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in avg_metrics.items()])
        self._logger.info(f"Average per horizon: {avg_str}")

        return avg_metrics

    def _save_evaluation_csv(
        self,
        avg_metrics: Dict[str, float],
        total_train_time: float,
    ):
        """Save evaluation metrics to CSV."""
        if self._stats_path is None:
            return

        eval_file = self._stats_path / "evaluation.csv"

        with open(eval_file, "w") as f:
            # Write header
            header = list(avg_metrics.keys()) + ["total_train_time"]
            f.write(",".join(header) + "\n")

            # Write values
            values = list(avg_metrics.values()) + [total_train_time]
            f.write(",".join([str(v) for v in values]) + "\n")

        if self._logger:
            self._logger.info(f"Evaluation results saved to {eval_file}")

    # ==========================================================================
    # Plotting Functions
    # ==========================================================================

    def _plot_test_results(self, preds: torch.Tensor, labels: torch.Tensor):
        """Generate test result plots."""
        if self._plot_path is None:
            return

        try:
            from src.utils.plotting import (
                plot_mean_per_day,
                mean_branch_plot,
                branch_plot,
            )

            # Mean predictions per horizon
            per_day_preds = [preds[:, :, i].mean() for i in range(self._pred_len)]
            per_day_labels = [labels[:, :, i].mean() for i in range(self._pred_len)]

            plot_mean_per_day(
                per_day_preds,
                per_day_labels,
                self._plot_path,
                "mean_per_day_performance_plot.png",
            )

            # Branch plots
            n_samples = min(preds.shape[0], 100)
            if n_samples > 0:
                mean_branch_plot(
                    preds[:n_samples, :, :],
                    labels[:n_samples, :, :],
                    self._plot_path,
                    "mean_performance_plot.png",
                )

                var_index = 0
                branch_plot(
                    preds[:n_samples, :, :],
                    labels[:n_samples, :, :],
                    var_index,
                    self._plot_path,
                    f"sensor_{var_index}_branch_plot.png",
                )

            self._logger.info("Test plots generated successfully")

        except Exception as e:
            if self._logger:
                self._logger.warning(f"Test plotting failed: {e}")

    # ==========================================================================
    # Training Functions
    # ==========================================================================

    def _forward_pass(self, h: int, X_df=None):
        """Execute forward pass (prediction) through the model."""
        return self._forecast_fn(h, X_df, None)

    def _fit_model(self, train_data, X_df=None):
        """Fit the model on training data."""
        if self._backend == "statsforecast":
            self.model.fit(train_data)
        else:
            self._fit_fn(train_data, X_df)

    # ==========================================================================
    # Main Training/Evaluation Methods
    # ==========================================================================

    def train(self) -> Optional[Dict[str, float]]:
        """
        Main training method.
        Delegates to appropriate method based on evaluation mode.
        """
        self._logger.info("Start training!")
        self._logger.info(f"Tracking metrics: {self._get_metric_names()}")

        if self._evaluation_mode == "single_split":
            return self._train_single_split_window()
        else:
            return self._train_multi_split()

    def _train_multi_split(self) -> Optional[Dict[str, float]]:
        """Multi split mode fit mode: fit once, evaluate on test."""
        train_data = self._get_train_data()

        # Fit the model
        t1 = time.time()
        self._fit_model(train_data)
        t2 = time.time()

        self._total_train_time = t2 - t1
        self._logger.info(f"Fitting completed in {self._total_train_time:.4f}s\n----------")

        # Save model
        if self._save_path:
            self.save_model(self._save_path)

        # Evaluate on test set
        test_result = self.evaluate("test")

        return test_result

    def _train_single_split_window(self) -> Optional[Dict[str, float]]:
        """Single Split window mode: fit on each window, collect all predictions."""
        from statsforecast import StatsForecast

        if self._dataloader_instance is None:
            raise ValueError("dataloader_instance required for single split window evaluation")

        all_preds = []
        all_labels = []

        # Get test loader for single split windows
        test_loader = self._dataloader_instance.get_test_sliding_loader(
            stride=self._single_split_stride,
            batch_size=1,
        )

        val_end = self._dataloader_instance.val_end

        self._logger.info(
            f"Evaluating with single split windows (stride={self._single_split_stride})"
        )

        t_start = time.time()

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing windows")):
            x_batch, y_batch = batch[0], batch[1]

            # Calculate time index for this window
            window_start_idx = val_end + batch_idx * self._single_split_stride

            # Get historical data in statsforecast format
            hist_long = self._get_historical_data_long(window_start_idx)

            # Create fresh model for each window
            if self._model_factory is not None:
                sf_model = self._model_factory()
                sf = StatsForecast(models=[sf_model], **self._sf_kwargs)
            else:
                # Reuse existing model (less ideal but fallback)
                sf = self.model

            # Fit and predict
            sf.fit(hist_long)
            forecast = sf.predict(h=self._pred_len)

            # Convert to tensor [1, channels, pred_len]
            pred_tensor = self._forecast_to_tensor(forecast, self._pred_len, self._num_channels)

            all_preds.append(pred_tensor)
            all_labels.append(y_batch)

        self._total_train_time = time.time() - t_start

        # Concatenate all predictions and labels
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        self._logger.info(f"Total windows evaluated: {len(all_preds)}")
        self._logger.info(f"Predictions shape: {preds.shape}, Labels shape: {labels.shape}")
        self._logger.info(f"Total training time: {self._total_train_time:.4f}s")

        # Compute and return test metrics
        return self._compute_test_metrics(preds, labels)

    def evaluate(self, mode: str) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            mode: 'val' or 'test'

        Returns:
            Dictionary of metric_name: value pairs
        """
        if self._evaluation_mode == "single_split":
            # Single split window already does evaluation during train
            self._logger.warning("Single split window mode evaluates during train()")
            return {}

        if mode == "test" and self._save_path:
            self.load_model(self._save_path)

        preds = []
        labels = []

        test_data = self._get_data(mode)

        # Generate predictions
        predictions = self._forward_pass(self._pred_len)

        # Convert predictions to tensor
        out_batch = statsforecast_to_tensor(predictions, self._alias, True)
        label = statsforecast_to_tensor(test_data, "y", True)

        preds.append(out_batch.cpu())
        labels.append(label.cpu())

        # Concatenate
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode == "val":
            preds, labels = self._prepare_test_data(preds, labels)
            return self._compute_metrics(preds, labels)
        elif mode == "test":
            return self._compute_test_metrics(preds, labels)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ==========================================================================
    # Direct API Methods (for compatibility)
    # ==========================================================================

    def fit(self, train_df, X_df=None):
        """Fit the model (direct API)."""
        self._fit_fn(train_df, X_df)
        return self

    def forecast(self, h: int, X_df=None, level=None):
        """Generate forecasts (direct API)."""
        return self._forecast_fn(h, X_df, level)
