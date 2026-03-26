from statsforecast.models import Naive

from src.base.nixtla_multi_split_experiment import NixtlaMultiSplitComparison
from src.base.nixtla_multi_split_experiment import (
    run_multi_split_experiment,
)
from src.utils.args import get_naive_multi_split_experiment_config


class NaiveTraining(NixtlaMultiSplitComparison):
    """Naive-specific training implementation."""

    def get_config_parser(self):
        return get_naive_multi_split_experiment_config()

    def get_model_name(self):
        return "naive"

    def create_statsforecast_model(self, args):
        return Naive(
            alias=args.alias,
        )


if __name__ == "__main__":
    run_multi_split_experiment(NaiveTraining)
