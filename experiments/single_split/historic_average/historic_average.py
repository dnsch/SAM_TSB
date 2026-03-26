from statsforecast.models import HistoricAverage

from src.base.nixtla_single_split_experiment import (
    NixtlaSingleSplitExperiment,
    run_nixtla_single_split_experiment,
)
from src.utils.args import get_historic_average_config


class HistoricAverageSingleSplitExperiment(NixtlaSingleSplitExperiment):
    """HistoricAverage model single split experiment implementation."""

    def get_config_parser(self):
        return get_historic_average_config()

    def get_model_name(self):
        return "historic_average"

    def create_statsforecast_model(self, args):
        return HistoricAverage(alias=args.alias)


if __name__ == "__main__":
    run_nixtla_single_split_experiment(HistoricAverageSingleSplitExperiment)
