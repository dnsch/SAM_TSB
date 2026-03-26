from statsforecast.models import Naive

from src.base.nixtla_single_split_experiment import (
    NixtlaSingleSplitExperiment,
    run_nixtla_single_split_experiment,
)
from src.utils.args import get_naive_config


class NaiveSingleSplitExperiment(NixtlaSingleSplitExperiment):
    """Naive model single split experiment implementation."""

    def get_config_parser(self):
        return get_naive_config()

    def get_model_name(self):
        return "naive"

    def create_statsforecast_model(self, args):
        return Naive(alias=args.alias)


if __name__ == "__main__":
    run_nixtla_single_split_experiment(NaiveSingleSplitExperiment)
