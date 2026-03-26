from statsforecast.models import HistoricAverage

from src.base.nixtla_multi_split_experiment import NixtlaMultiSplitComparison
from src.base.nixtla_multi_split_experiment import (
    run_multi_split_experiment,
)
from src.utils.args import get_historic_average_multi_split_experiment_config


class HistoricAverageTraining(NixtlaMultiSplitComparison):
    """HistoricAverage-specific training implementation."""

    def get_config_parser(self):
        return get_historic_average_multi_split_experiment_config()

    def get_model_name(self):
        return "historic_average"

    def create_statsforecast_model(self, args):
        return HistoricAverage(
            alias=args.alias,
        )


if __name__ == "__main__":
    run_multi_split_experiment(HistoricAverageTraining)
