from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)

# from src.models.time_series.tsmixer.tsmixer_ext import TSMixerExt
from src.models.time_series.tsmixer import TSMixerExt
from src.engines.tsmixer_ext_engine import TSMixerExt_Engine
from src.utils.args import get_tsmixer_ext_config


class TSMixerExtExperiment(TorchSingleSplitExperiment):
    """TSMixerExt-specific training implementation."""

    def __init__(self):
        super().__init__()
        self._extra_channels = None
        self._static_channels = None

    def get_config_parser(self):
        return get_tsmixer_ext_config()

    def get_model_name(self):
        return "tsmixer_ext"

    def get_engine_class(self):
        return TSMixerExt_Engine

    def get_metrics(self):
        return ["mape", "rmse"]

    def get_dataloader_kwargs(self, args):
        """Override to enable time features for TSMixerExt."""
        kwargs = super().get_dataloader_kwargs(args)
        # needs special handling with the time features
        # TODO: make that cleaner
        kwargs.update(
            {
                "model_type": "tsmixer_ext",
            }
        )
        return kwargs

    def create_model(self, args, dataloader):
        """Create TSMixerExt model instance."""

        # automatic retrieval of input_channels, extra_channels, and
        # output_channels if these are not set manually

        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        self._extra_channels = self.dataloader_instance.get_extra_channels(
            getattr(args, "extra_channels", None)
        )

        self._static_channels = self.dataloader_instance.get_static_channels(
            getattr(args, "static_channels", None)
        )

        return TSMixerExt(
            input_channels=self._input_channels,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            extra_channels=self._extra_channels,
            hidden_channels=args.hidden_channels,
            static_channels=self._static_channels,
            activation_fn=args.activation,
            num_blocks=args.num_blocks,
            dropout_rate=args.dropout_rate,
            ff_dim=args.ff_dim,
            normalize_before=args.normalize_before,
            norm_type=args.norm_type,
        )


if __name__ == "__main__":
    run_single_split_experiment(TSMixerExtExperiment)
