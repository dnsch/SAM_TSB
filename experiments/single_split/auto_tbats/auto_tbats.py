from statsforecast.models import AutoTBATS

from src.base.nixtla_standard_experiment import (
    NixtlaSingleSplitExperiment,
    run_nixtla_single_split_experiment,
)
from src.utils.args import get_auto_tbats_config


import warnings

warnings.filterwarnings("ignore")


class AutoTBATSSingleSplitExperiment(NixtlaSingleSplitExperiment):
    """AutoTBATS model single split experiment implementation."""

    def get_config_parser(self):
        return get_auto_tbats_config()

    def get_model_name(self):
        return "autotbats"

    def create_statsforecast_model(self, args):
        return AutoTBATS(
            season_length=args.season_length,
            use_boxcox=args.use_boxcox,
            bc_lower_bound=args.bc_lower_bound,
            bc_upper_bound=args.bc_upper_bound,
            use_trend=args.use_trend,
            use_damped_trend=args.use_damped_trend,
            use_arma_errors=args.use_arma_errors,
            alias=args.alias,
        )


if __name__ == "__main__":
    run_nixtla_single_split_experiment(AutoTBATSSingleSplitExperiment)
