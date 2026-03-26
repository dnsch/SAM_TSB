from statsforecast.models import AutoMFLES

from src.base.nixtla_multi_split_experiment import NixtlaMultiSplitComparison
from src.base.nixtla_multi_split_experiment import (
    run_multi_split_experiment,
)
from src.utils.args import get_automfles_multi_split_experiment_config


class AutoMFLESTraining(NixtlaMultiSplitComparison):
    """AutoMFLES-specific training implementation."""

    def get_config_parser(self):
        return get_automfles_multi_split_experiment_config()

    def get_model_name(self):
        return "automfles"

    def create_statsforecast_model(self, args):
        return AutoMFLES(
            test_size=args.pred_len,
            season_length=args.season_length,
            n_windows=args.n_windows,
            metric=args.metric,
            verbose=args.verbose,
            prediction_intervals=args.prediction_intervals,
        )


if __name__ == "__main__":
    run_multi_split_experiment(AutoMFLESTraining)
