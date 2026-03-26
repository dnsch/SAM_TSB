from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import torch

# Configure matplotlib for better-looking plots

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))

from src.utils.metrics import TrainingMetrics

# ============================================================

# Plots

# ============================================================


def plot_loss_metric(
    train_values: List[float],
    val_values: List[float],
    metric_name: str,
    epochs: int,
    plot_path: Union[str, Path] = ".",
    figsize=(10, 6),
):
    """
    Plot training and validation curves for a single metric.

    Args:
        train_values: List of training metric values per epoch
        val_values: List of validation metric values per epoch
        metric_name: Name of the metric (e.g., 'mae', 'rmse', 'mape')
        epochs: Total number of epochs
        plot_path: Base directory for saving plots
        figsize: Figure size as (width, height)
        show_best: Whether to mark the best validation value on the plot
    """
    plot_dir = Path(plot_path) / "statistics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    epoch_range = np.arange(1, epochs + 1)

    # Plot training curve
    ax.plot(
        epoch_range[: len(train_values)],
        train_values,
        label="Training",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, epochs // 20),
        alpha=0.9,
    )

    # Plot validation curve
    ax.plot(
        epoch_range[: len(val_values)],
        val_values,
        label="Validation",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
        markevery=max(1, epochs // 20),
        alpha=0.9,
    )

    # Styling
    ax.set_xlabel("Epoch", fontweight="medium")
    ax.set_ylabel("Value", fontweight="medium")

    # Format title
    title = metric_name.replace("_", " ").upper()
    ax.set_title(title, fontweight="bold", pad=15)

    # Grid
    ax.grid(True, linestyle=":", alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="best", framealpha=0.95, edgecolor="gray")

    # Tight layout
    plt.tight_layout()

    # Save
    plot_filename = plot_dir / f"train_val_{metric_name}_{epochs}epochs.png"
    plt.savefig(plot_filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stats(
    metrics: TrainingMetrics,
    last_epoch: int,
    plot_path: Path,
):
    """
    Generic plotting function for training metrics.

    Args:
        metrics: TrainingMetrics object containing all tracked metrics
        loss_name: Name of the loss metric used for optimization
        last_epoch: Last epoch number for x-axis
        plot_path: Path to save plots
    """
    all_metric_names = sorted(metrics.get_all_metric_names())

    # Plot combined train/val curves for all metrics
    for metric_name in all_metric_names:
        train_values = metrics.get_train_metric(metric_name)
        val_values = metrics.get_val_metric(metric_name)

        if train_values and val_values:
            plot_loss_metric(
                train_values=train_values,
                val_values=val_values,
                metric_name=metric_name,
                epochs=last_epoch,
                plot_path=plot_path,
            )


def plot_all_metrics_combined(
    metrics: TrainingMetrics,
    last_epoch: int,
    plot_path: Path,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Plot all metrics stacked vertically in a single column.

    Args:
        metrics: TrainingMetrics object containing all tracked metrics
        last_epoch: Last epoch number
        plot_path: Path to save the plot
        figsize: Optional figure size (width, height). If None, auto-calculated.
    """

    all_metrics = sorted(metrics.get_all_metric_names())
    n_metrics = len(all_metrics)

    if n_metrics == 0:
        return

    # Vertical stacking: 1 column, n rows
    n_cols = 1
    n_rows = n_metrics

    # Auto-calculate figure size if not provided
    # Single column, so narrow width; height scales with number of metrics
    if figsize is None:
        figsize = (10, 4 * n_metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single metric case
    axes = [axes] if n_metrics == 1 else axes.flatten()

    epochs = list(range(1, last_epoch + 1))

    for idx, metric_name in enumerate(all_metrics):
        ax = axes[idx]

        train_vals = metrics.get_train_metric(metric_name)
        val_vals = metrics.get_val_metric(metric_name)

        if train_vals:
            ax.plot(
                epochs[: len(train_vals)],
                train_vals,
                label="Train",
                color="#2E86AB",
                linewidth=2,
                marker="o",
                markersize=4,
                markevery=max(1, last_epoch // 20),
                alpha=0.9,
            )
        if val_vals:
            ax.plot(
                epochs[: len(val_vals)],
                val_vals,
                label="Validation",
                color="#F77F00",
                linewidth=2,
                linestyle="--",
                marker="s",
                markersize=4,
                markevery=max(1, last_epoch // 20),
                alpha=0.9,
            )

        ax.set_title(f"{metric_name.upper()}", fontweight="bold", pad=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.upper())
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, linestyle=":", alpha=0.6, linewidth=0.8)
        ax.set_axisbelow(True)

    # Add overall title
    fig.suptitle(
        "Training Metrics Overview",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(
        plot_path / f"all_metrics_combined_{last_epoch}ep.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def branch_plot(preds, labels, var_index, plot_path, title):
    # Select the specified var_index
    preds_sensor = preds[:, var_index, :]
    labels_sensor = labels[:, var_index, :]
    # Define a set of distinguishable colors
    colors = ["orange", "green", "red", "purple", "brown", "pink"]

    # Create a plot
    plt.figure(figsize=(15, 7), dpi=500)

    for timepoint_idx in range(labels_sensor.shape[0]):
        labels_entry = labels_sensor[timepoint_idx, :]
        preds_entry = preds_sensor[timepoint_idx, :]

        plt.plot(
            range(timepoint_idx, len(labels_entry) + timepoint_idx),
            labels_entry,
            color="blue",
            linewidth=0.5,
            label="Ground Truth" if timepoint_idx == 0 else "",
        )
        plt.plot(
            range(timepoint_idx, len(preds_entry) + timepoint_idx),
            preds_entry,
            color=colors[timepoint_idx % len(colors)],
            linewidth=0.5,
            label=f"Prediction {timepoint_idx + 1}" if timepoint_idx == 0 else "",
        )

    # Adding labels, title, and legend
    plt.title(f"Ground Truth and Predictions for Sensor {var_index}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid()
    # Create custom legend entries
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, label="Ground Truth"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1,
            label=f"Predictions starting from first {preds.shape[0]} timesteps",
            linestyle="-",
            marker=None,
        ),
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_dir / "statistics" / title
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)


def mean_branch_plot(preds, labels, plot_path, title):
    # Select the specified sensor
    preds_sensor = preds.mean(dim=1)
    labels_sensor = labels.mean(dim=1)
    # Define a set of distinguishable colors
    colors = ["orange", "green", "red", "purple", "brown", "pink"]

    # Create a plot
    plt.figure(figsize=(15, 7), dpi=500)

    # Plot all sequences for the specified sensor

    for timepoint_idx in range(labels_sensor.shape[0]):
        labels_entry = labels_sensor[timepoint_idx, :]
        preds_entry = preds_sensor[timepoint_idx, :]
        # labels_entry = labels_sensor[:, timepoint_idx]
        # preds_entry = preds_sensor[:, timepoint_idx]

        plt.plot(
            range(timepoint_idx, len(labels_entry) + timepoint_idx),
            labels_entry,
            color="blue",
            linewidth=0.5,
            label="Ground Truth" if timepoint_idx == 0 else "",
        )
        plt.plot(
            range(timepoint_idx, len(preds_entry) + timepoint_idx),
            preds_entry,
            color=colors[timepoint_idx % len(colors)],
            linewidth=0.5,
            label=f"Prediction {timepoint_idx + 1}" if timepoint_idx == 0 else "",
        )

    # Adding labels, title, and legend
    plt.title(f"Mean Ground Truth and Predictions for all Sensors")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid()
    # Create custom legend entries
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1, label="Ground Truth"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1,
            label=f"Predictions starting from first {preds.shape[0]} timesteps",
            linestyle="-",
            marker=None,
        ),
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    # Save the plot
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_dir / "statistics" / title
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)


def plot_mean_per_day(mean_per_day_preds, mean_per_day_labels, plot_path, title):
    # Convert tensors to lists
    preds = [p.item() for p in mean_per_day_preds]
    labels = [l.item() for l in mean_per_day_labels]

    # Create a plot
    plt.figure(figsize=(15, 7), dpi=500)

    # Plot predictions and labels
    plt.plot(range(len(labels)), labels, color="blue", linewidth=0.5, label="Ground Truth")
    plt.plot(range(len(preds)), preds, color="orange", linewidth=0.5, label="Predictions")

    # Adding labels, title, and legend
    plt.title("Timestep Mean Labels and Predictions for all Sensors")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid()
    plt.legend(loc="upper right")
    # Save the plot
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_dir / "statistics" / title
    plot_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)


def plot_dataset_splits(
    df_raw: pd.DataFrame,
    dataset_name: str,
    train_end: int,
    val_end: int,
    test_end=None,
    save_dir="results/datasets",
    figsize=(14, 5),
    dpi=150,
    save_mean_csv: bool = True,
):
    """
    Plot each variable in the dataset with train/val/test split visualization.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe with datetime index and variable columns
    dataset_name : str
        Name of the dataset (used for folder naming)
    train_end : int
        Index where training data ends
    val_end : int
        Index where validation data ends
    test_end : int, optional
        Index where test data ends (defaults to len(df_raw))
    save_dir : str
        Base directory for saving plots
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution for saved figures
    """

    # Colors matching your training plots
    TRAIN_COLOR = "#ff7f0e"  # Orange
    VAL_COLOR = "#2ca02c"  # Green
    TEST_COLOR = "#9467bd"  # Purple

    if test_end is None:
        test_end = len(df_raw)

    # Create output directory
    output_dir = Path(save_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure datetime index
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw = df_raw.copy()
        df_raw.index = pd.to_datetime(df_raw.index)

    if save_mean_csv:
        save_mean_data_splits(
            df_raw=df_raw,
            dataset_name=dataset_name,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
            save_dir="datasets/mean_data",  # Fixed path as you specified
        )

    # Define split colors and labels
    splits = [
        (0, train_end, "Train", TRAIN_COLOR, 0.3),
        (train_end, val_end, "Validation", VAL_COLOR, 0.3),
        (val_end, test_end, "Test", TEST_COLOR, 0.3),
    ]

    # Check number of variables and ask user if > 10
    n_vars = len(df_raw.columns)
    plot_all_variables = True

    if n_vars > 10:
        print(f"\nDataset contains {n_vars} variables.")
        while True:
            response = (
                input(f"Do you want to plot all {n_vars} variables? (yes/no): ").strip().lower()
            )
            if response in ["yes", "y"]:
                plot_all_variables = True
                print("Plotting all variables + mean...")
                break
            elif response in ["no", "n"]:
                plot_all_variables = False
                print("Plotting only the mean over all variables...")
                break
            else:
                print("Please enter 'yes' or 'no'.")

    # Helper function to create a single variable plot
    def plot_single_variable(data_series, var_name, filename, ylabel=None):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot background shading for splits
        for start, end, label, color, alpha in splits:
            ax.axvspan(
                df_raw.index[start],
                df_raw.index[min(end, len(df_raw)) - 1],
                alpha=alpha,
                color=color,
                label=f"{label} (n={end - start})",
            )

        # Plot the actual data
        ax.plot(
            df_raw.index[:test_end], data_series[:test_end], color="black", linewidth=0.5, alpha=0.8
        )

        # Add vertical lines at split boundaries
        for idx, color in [(train_end, TRAIN_COLOR), (val_end, VAL_COLOR)]:
            ax.axvline(df_raw.index[idx], color=color, linestyle="--", linewidth=2, alpha=0.8)

        # Formatting
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ylabel if ylabel else var_name, fontsize=12)
        ax.set_title(
            f"{dataset_name} - {var_name}\n"
            f"Train: {train_end} | Val: {val_end - train_end} | Test: {test_end - val_end} samples",
            fontsize=14,
            fontweight="bold",
        )

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        # Legend
        ax.legend(loc="upper right", fontsize=10)

        # Grid
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        # Tight layout
        plt.tight_layout()

        # Save
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"Saved: {save_path}")

    # Plot each variable individually (only if user agreed or <= 10 variables)
    if plot_all_variables:
        for col in df_raw.columns:
            plot_single_variable(df_raw[col].values, col, f"{col}_splits.png")

    # Plot mean time series over all variables (always plotted)
    mean_series = df_raw.mean(axis=1).values
    plot_single_variable(
        mean_series, "Mean (All Variables)", "mean_all_variables_splits.png", ylabel="Mean Value"
    )

    # Create summary plot with all variables (only if user agreed or <= 10 variables)
    if plot_all_variables:
        fig, axes = plt.subplots(n_vars + 1, 1, figsize=(figsize[0], 3 * (n_vars + 1)), sharex=True)

        # Plot individual variables
        for ax, col in zip(axes[:-1], df_raw.columns):
            # Background shading
            for start, end, label, color, alpha in splits:
                ax.axvspan(
                    df_raw.index[start],
                    df_raw.index[min(end, len(df_raw)) - 1],
                    alpha=alpha,
                    color=color,
                )

            # Data line
            ax.plot(
                df_raw.index[:test_end], df_raw[col].values[:test_end], color="black", linewidth=0.5
            )

            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3)

        # Plot mean in the last subplot
        ax_mean = axes[-1]
        for start, end, label, color, alpha in splits:
            ax_mean.axvspan(
                df_raw.index[start],
                df_raw.index[min(end, len(df_raw)) - 1],
                alpha=alpha,
                color=color,
            )

        ax_mean.plot(df_raw.index[:test_end], mean_series[:test_end], color="black", linewidth=0.5)
        ax_mean.set_ylabel("Mean", fontsize=10, fontweight="bold")
        ax_mean.grid(True, alpha=0.3)

        # Common x-label and formatting
        axes[-1].set_xlabel("Date", fontsize=12)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=TRAIN_COLOR, alpha=0.3, label=f"Train (n={train_end})"),
            Patch(facecolor=VAL_COLOR, alpha=0.3, label=f"Val (n={val_end - train_end})"),
            Patch(facecolor=TEST_COLOR, alpha=0.3, label=f"Test (n={test_end - val_end})"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=10
        )

        # Title
        fig.suptitle(
            f"{dataset_name} - All Variables + Mean", fontsize=14, fontweight="bold", y=1.01
        )

        plt.tight_layout()

        save_path = output_dir / "all_variables_splits.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"Saved: {save_path}")

    # Print split statistics
    print(f"\n{'=' * 60}")
    print(f"Split Statistics for {dataset_name}")
    print(f"{'=' * 60}")

    stats_data = []
    for split_name, start, end in [
        ("Train", 0, train_end),
        ("Val", train_end, val_end),
        ("Test", val_end, test_end),
    ]:
        split_df = df_raw.iloc[start:end]
        stats_data.append(
            {
                "Split": split_name,
                "Start": df_raw.index[start].strftime("%Y-%m-%d"),
                "End": df_raw.index[end - 1].strftime("%Y-%m-%d"),
                "Samples": end - start,
                "Mean": split_df.values.mean(),
                "Std": split_df.values.std(),
            }
        )

    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    print(f"{'=' * 60}")
    print(f"\nAll plots saved to: {output_dir.absolute()}")


PLOT_STYLE_CONFIG = {
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2,
    "axes.linewidth": 1.2,
}

# Colors

GROUND_TRUTH_COLOR = "#1f77b4"  # Blue
PREDICTION_COLOR = "#E74C3C"  # Red

# Alpha values (ground truth less prominent)

GROUND_TRUTH_LINE_ALPHA = 0.45
GROUND_TRUTH_FILL_ALPHA = 0.15
PREDICTION_LINE_ALPHA = 0.9
PREDICTION_FILL_ALPHA = 0.25


def plot_mean_forecasts(
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
    ground_truth: np.ndarray,
    plot_path: str,
    pred_len: Optional[int] = None,
    title_prefix: str = "mean_forecast",
    figsize: Tuple[int, int] = (20, 8),
    dpi: int = 300,
    show_std: bool = True,
    show_percent: float = 100.0,
    show_steps: Optional[int] = None,
    plot_all_variables: bool = False,
    save_forecast_csv: bool = True,
) -> None:
    """
    Plot mean forecasts and ground truth.

    Creates:
    1. One plot with all variables averaged (always)
    2. One plot per variable (only if plot_all_variables=True)
    3. CSV file with forecast values (if save_forecast_csv=True)

    Args:
        mean_preds: shape [total_length, num_variables]
        std_preds: shape [total_length, num_variables]
        ground_truth: shape [total_length, num_variables]
        plot_path: Base path for saving plots
        pred_len: Prediction length (used for show_steps default and CSV filename)
        title_prefix: Prefix for plot filenames
        figsize: Figure size (width, height)
        dpi: Resolution for saved figures
        show_std: Whether to show standard deviation bands for predictions
        show_percent: Percentage of data to show (0-100). Default 100 shows all data.
                      Ignored if show_steps or pred_len is provided.
        show_steps: Exact number of steps to show. If provided, overrides show_percent.
                    If None and pred_len is provided, uses pred_len.
        plot_all_variables: If True, also plot each variable separately. Default False.
        save_forecast_csv: If True, save forecast values to CSV. Default True.
    """
    # Apply style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE_CONFIG)

    total_length = mean_preds.shape[0]
    num_variables = mean_preds.shape[1]

    # Use pred_len as default for show_steps if not explicitly provided
    if show_steps is None and pred_len is not None:
        show_steps = pred_len

    # Calculate cutoff index
    if show_steps is not None:
        # Use absolute number of steps
        cutoff_idx = max(1, min(show_steps, total_length))
        cutoff_suffix = f"_{cutoff_idx}steps"
        cutoff_title = f" (first {cutoff_idx} steps)"
    else:
        # Use percentage
        show_percent = np.clip(show_percent, 0.0, 100.0)
        cutoff_idx = max(1, int(total_length * show_percent / 100.0))
        cutoff_suffix = f"_{int(show_percent)}pct" if show_percent < 100.0 else ""
        cutoff_title = f" (first {show_percent:.1f}%)" if show_percent < 100.0 else ""

    # Slice arrays to show only the requested portion
    mean_preds_sliced = mean_preds[:cutoff_idx]
    std_preds_sliced = std_preds[:cutoff_idx]
    ground_truth_sliced = ground_truth[:cutoff_idx]

    timesteps = np.arange(cutoff_idx)

    # Create output directories
    plot_dir = Path(plot_path) / "statistics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    forecast_dir = Path(plot_path).parent / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    plots_saved = 0

    # Save forecast CSV
    if save_forecast_csv:
        csv_pred_len = pred_len if pred_len is not None else cutoff_idx
        csv_filename = forecast_dir / f"{csv_pred_len}_step_forecast.csv"
        _save_forecast_csv(
            mean_preds=mean_preds,
            std_preds=std_preds,
            ground_truth=ground_truth,
            filepath=csv_filename,
        )
        print(f"Saved forecast CSV to {csv_filename}")

    # Plot for each variable (only if requested)
    if plot_all_variables:
        for var_idx in range(num_variables):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Plot prediction std band first (so it's behind the lines)
            if show_std:
                ax.fill_between(
                    timesteps,
                    mean_preds_sliced[:, var_idx] - std_preds_sliced[:, var_idx],
                    mean_preds_sliced[:, var_idx] + std_preds_sliced[:, var_idx],
                    color=PREDICTION_COLOR,
                    alpha=PREDICTION_FILL_ALPHA,
                )

            # Plot ground truth line (less prominent)
            ax.plot(
                timesteps,
                ground_truth_sliced[:, var_idx],
                color=GROUND_TRUTH_COLOR,
                linewidth=2,
                linestyle="-",
                alpha=GROUND_TRUTH_LINE_ALPHA,
            )
            # Plot prediction line (more prominent)
            ax.plot(
                timesteps,
                mean_preds_sliced[:, var_idx],
                color=PREDICTION_COLOR,
                linewidth=2.5,
                linestyle="--",
                alpha=PREDICTION_LINE_ALPHA,
            )

            ax.set_xlabel("Time Step $t$", fontsize=18, fontweight="bold")
            ax.set_ylabel("Value", fontsize=18, fontweight="bold")
            ax.set_xlim(-cutoff_idx * 0.02, cutoff_idx * 1.02)

            # Legend
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color=GROUND_TRUTH_COLOR,
                    linewidth=2,
                    linestyle="-",
                    alpha=GROUND_TRUTH_LINE_ALPHA,
                    label="Ground Truth",
                ),
                Line2D(
                    [0],
                    [0],
                    color=PREDICTION_COLOR,
                    linewidth=2.5,
                    linestyle="--",
                    alpha=PREDICTION_LINE_ALPHA,
                    label="Prediction (mean)",
                ),
            ]
            if show_std:
                legend_elements.append(
                    mpatches.Patch(
                        facecolor=PREDICTION_COLOR,
                        alpha=PREDICTION_FILL_ALPHA,
                        edgecolor=PREDICTION_COLOR,
                        label="Prediction ±1σ",
                    )
                )

            fig.legend(
                handles=legend_elements,
                loc="lower center",
                ncol=3 if show_std else 2,
                fontsize=14,
                framealpha=0.95,
                bbox_to_anchor=(0.5, 0.01),
                handlelength=3,
                handleheight=1.5,
            )

            # Title
            title = f"Mean Forecast – Variable {var_idx}{cutoff_title}"
            plt.suptitle(title, fontsize=20, fontweight="bold", y=0.995)

            plt.tight_layout(rect=[0, 0.08, 1.0, 0.96])

            filename = plot_dir / f"{title_prefix}_var_{var_idx}{cutoff_suffix}.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            plots_saved += 1

    # Plot mean over all variables (always)
    overall_mean_preds = mean_preds_sliced.mean(axis=1)
    overall_std_preds = std_preds_sliced.mean(axis=1)
    overall_ground_truth = ground_truth_sliced.mean(axis=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot prediction std band first
    if show_std:
        ax.fill_between(
            timesteps,
            overall_mean_preds - overall_std_preds,
            overall_mean_preds + overall_std_preds,
            color=PREDICTION_COLOR,
            alpha=PREDICTION_FILL_ALPHA,
        )

    # Plot lines
    ax.plot(
        timesteps,
        overall_ground_truth,
        color=GROUND_TRUTH_COLOR,
        linewidth=2,
        linestyle="-",
        alpha=GROUND_TRUTH_LINE_ALPHA,
    )
    ax.plot(
        timesteps,
        overall_mean_preds,
        color=PREDICTION_COLOR,
        linewidth=2.5,
        linestyle="--",
        alpha=PREDICTION_LINE_ALPHA,
    )

    ax.set_xlabel("Time Step $t$", fontsize=18, fontweight="bold")
    ax.set_ylabel("Value", fontsize=18, fontweight="bold")
    ax.set_xlim(-cutoff_idx * 0.02, cutoff_idx * 1.02)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=GROUND_TRUTH_COLOR,
            linewidth=2,
            linestyle="-",
            alpha=GROUND_TRUTH_LINE_ALPHA,
            label="Ground Truth",
        ),
        Line2D(
            [0],
            [0],
            color=PREDICTION_COLOR,
            linewidth=2.5,
            linestyle="--",
            alpha=PREDICTION_LINE_ALPHA,
            label="Prediction (mean)",
        ),
    ]
    if show_std:
        legend_elements.append(
            mpatches.Patch(
                facecolor=PREDICTION_COLOR,
                alpha=PREDICTION_FILL_ALPHA,
                edgecolor=PREDICTION_COLOR,
                label="Prediction ±1σ",
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3 if show_std else 2,
        fontsize=14,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=3,
        handleheight=1.5,
    )

    title = f"Mean Forecast – All Variables Averaged{cutoff_title}"
    plt.suptitle(title, fontsize=20, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0.08, 1.0, 0.96])

    filename = plot_dir / f"{title_prefix}_all_variables{cutoff_suffix}.png"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    plots_saved += 1

    print(f"Saved {plots_saved} plot(s) to {plot_dir}")
    print(f"  Showing {cutoff_idx}/{total_length} timesteps")


# TODO: move to functions.py or another file
#
def save_mean_data_splits(
    df_raw: pd.DataFrame,
    dataset_name: str,
    train_end: int,
    val_end: int,
    test_end: int = None,
    save_dir: str = "datasets/mean_data",
) -> Path:
    """
    Save mean data with train/val/test split labels to a CSV file.
    """
    if test_end is None:
        test_end = len(df_raw)

    # Create output directory
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure datetime index
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw = df_raw.copy()
        df_raw.index = pd.to_datetime(df_raw.index)

    # Get number of variables
    num_variables = len(df_raw.columns)

    # Compute mean over all variables
    mean_series = df_raw.mean(axis=1)

    # Create split labels
    split_labels = pd.Series(index=df_raw.index, dtype=str)
    split_labels.iloc[:train_end] = "train"
    split_labels.iloc[train_end:val_end] = "val"
    split_labels.iloc[val_end:test_end] = "test"

    # Create output DataFrame
    output_df = pd.DataFrame(
        {
            "datetime": df_raw.index[:test_end],
            "mean_value": mean_series.iloc[:test_end].values,
            "split": split_labels.iloc[:test_end].values,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
            "num_variables": num_variables,  # NEW
        }
    )

    # Save to CSV
    filename = f"{dataset_name.lower()}_var_mean.csv"
    save_path = output_dir / filename
    output_df.to_csv(save_path, index=False)

    print(f"Saved mean data splits to: {save_path}")

    return save_path


#
#
#
def _save_forecast_csv(
    mean_preds: np.ndarray,
    std_preds: np.ndarray,
    ground_truth: np.ndarray,
    filepath: Path,
) -> None:
    """
    Save forecast values and ground truth to CSV.

    Args:
        mean_preds: shape [total_length, num_variables]
        std_preds: shape [total_length, num_variables]
        ground_truth: shape [total_length, num_variables]
        filepath: Path to save the CSV file
    """
    import pandas as pd

    total_length = mean_preds.shape[0]
    num_variables = mean_preds.shape[1]

    # Build DataFrame
    data = {"timestep": np.arange(total_length)}

    for var_idx in range(num_variables):
        data[f"ground_truth_var_{var_idx}"] = ground_truth[:, var_idx]
        data[f"mean_pred_var_{var_idx}"] = mean_preds[:, var_idx]
        data[f"std_pred_var_{var_idx}"] = std_preds[:, var_idx]

    # Add overall averages
    data["ground_truth_mean"] = ground_truth.mean(axis=1)
    data["mean_pred_mean"] = mean_preds.mean(axis=1)
    data["std_pred_mean"] = std_preds.mean(axis=1)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
