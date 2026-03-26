from pathlib import Path

# =============================================================================

# Project Structure

# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_dir(path: Path):
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Core directories (created on module load)

DATA_DIR = _ensure_dir(PROJECT_ROOT / "data")
RESULTS_DIR = _ensure_dir(PROJECT_ROOT / "results")


# =============================================================================

# Data Paths

# =============================================================================


def get_data_dir():
    """Get absolute path to data directory."""
    return DATA_DIR


def get_data_path(filename: str, must_exist=True):
    """Get absolute path to a data file."""
    path = DATA_DIR / filename
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def get_autoformer_dataset_path(dataset: str):
    """Get absolute path to autoformer dataset."""
    path = DATA_DIR / "autoformer_datasets" / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Please ensure that you have downloaded the data via the "
            "scripts/download_autoformer_dataset.py script"
        )
    return path


# =============================================================================

# Results Paths

# =============================================================================


def get_results_dir():
    """Get absolute path to results directory."""
    return RESULTS_DIR


def get_results_path(experiment: str):
    """Get absolute path to results folder (creates subdirectory)."""
    return _ensure_dir(RESULTS_DIR / experiment)


def get_experiment_results_dir(results_subdir: str):
    """
    Get the base results directory for an experiment type.

    Args:
        results_subdir: Subdirectory name under results/
                       (e.g., 'single_split', 'multi_split')

    Returns:
        Path to results/results_subdir/
    """
    return _ensure_dir(RESULTS_DIR / results_subdir)


def get_experiment_output_path(experiment: str, filename: str):
    """Get path for saving experiment output."""
    return get_results_path(experiment) / filename
