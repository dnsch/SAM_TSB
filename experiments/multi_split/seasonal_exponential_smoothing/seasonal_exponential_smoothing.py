from statsforecast.models import SeasonalExponentialSmoothingOptimized

from src.base.nixtla_multi_split_experiment import NixtlaMultiSplitComparison
from src.base.nixtla_multi_split_experiment import (
    run_multi_split_experiment,
)
from src.utils.args import get_seasonal_exponential_smoothing_multi_split_experiment_config


class SeasonalExponentialSmoothingTraining(NixtlaMultiSplitComparison):
    """SeasonalExponentialSmoothingOptimized-specific training implementation."""

    def get_config_parser(self):
        return get_seasonal_exponential_smoothing_multi_split_experiment_config()

    def get_model_name(self):
        return "seasonal_exponential_smoothing"

    def create_statsforecast_model(self, args):
        return SeasonalExponentialSmoothingOptimized(
            season_length=args.season_length,
            alias=args.alias,
        )


if __name__ == "__main__":
    run_multi_split_experiment(SeasonalExponentialSmoothingTraining)
