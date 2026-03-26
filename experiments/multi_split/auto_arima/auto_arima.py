from statsforecast.models import AutoARIMA

from src.base.nixtla_multi_split_experiment import NixtlaMultiSplitComparison
from src.base.nixtla_multi_split_experiment import (
    run_multi_split_experiment,
)
from src.utils.args import get_auto_arima_multi_split_experiment_config


class AutoARIMATraining(NixtlaMultiSplitComparison):
    """AutoARIMA-specific training implementation."""

    def get_config_parser(self):
        return get_auto_arima_multi_split_experiment_config()

    def get_model_name(self):
        return "autoarima"

    def create_statsforecast_model(self, args):
        return AutoARIMA(
            season_length=args.seasonal_periods,
            max_p=args.max_p,
            max_q=args.max_q,
            max_P=args.max_P,
            max_Q=args.max_Q,
            d=args.d,
            D=args.D,
            stepwise=True,
            approximation=True,
            seasonal=args.seasonal,
            ic="aic",
        )


if __name__ == "__main__":
    run_multi_split_experiment(AutoARIMATraining)
