from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.moment import MOMENT
from src.engines.moment_engine import MOMENT_Engine
from src.utils.args import get_moment_config


class MOMENTExperiment(TorchSingleSplitExperiment):
    """MOMENT-specific training implementation."""

    def get_config_parser(self):
        return get_moment_config()

    def get_model_name(self):
        return "moment"

    def get_engine_class(self):
        return MOMENT_Engine

    def get_metrics(self):
        return ["mse", "mae", "mape", "rmse"]

    def get_engine_kwargs(
        self, args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
    ):
        """Override to add MOMENT-specific engine parameters."""
        kwargs = super().get_engine_kwargs(
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
        )
        # Add log interval for progress tracking
        kwargs["log_interval"] = getattr(args, "log_interval", 50)
        return kwargs

    def create_model(self, args, dataloader):
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        return MOMENT(
            input_channels=self._input_channels,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            model_name=getattr(args, "moment_model_name", "AutonLab/MOMENT-1-large"),
            head_dropout=getattr(args, "head_dropout", 0.1),
            weight_decay=getattr(args, "wdecay", 0),
            freeze_encoder=getattr(args, "freeze_encoder", True),
            freeze_embedder=getattr(args, "freeze_embedder", True),
            freeze_head=getattr(args, "freeze_head", False),
        )


if __name__ == "__main__":
    run_single_split_experiment(MOMENTExperiment)
