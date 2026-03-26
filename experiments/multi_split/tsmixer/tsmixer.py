from src.base.torch_multi_split_experiment import (
    TorchMultiSplitExperiment,
    run_multi_split_experiment,
)
from src.models.time_series.tsmixer import TSMixer
from src.engines.tsmixer_engine import TSMixer_Engine
from src.utils.args import get_tsmixer_multi_split_experiment_config


class TSMixerMultiSplitExperiment(TorchMultiSplitExperiment):
    """TSMixer multi split experiment implementation."""

    def get_config_parser(self):
        return get_tsmixer_multi_split_experiment_config()

    def get_model_name(self):
        return "tsmixer"

    def get_engine_class(self):
        return TSMixer_Engine

    def get_metrics(self):
        """Override to specify TSMixer metrics."""
        return ["mse", "mae", "mape", "rmse"]

    def get_engine_kwargs(
        self,
        args,
        model,
        dataloader,
        scaler,
        optimizer,
        scheduler,
        loss_fn,
        log_dir,
        logger,
        split_idx=0,
    ):
        """Override to add TSMixer-specific engine parameters."""
        kwargs = super().get_engine_kwargs(
            args,
            model,
            dataloader,
            scaler,
            optimizer,
            scheduler,
            loss_fn,
            log_dir,
            logger,
            split_idx,
        )
        return kwargs

    def create_model(self, args, dataloader):
        """Create TSMixer model instance."""
        # Determine output channels (default to input channels if not specified)
        output_channels = args.output_channels if args.output_channels else self._input_channels

        return TSMixer(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            input_channels=self._input_channels,
            output_channels=output_channels,
            activation=args.activation,
            num_blocks=args.num_blocks,
            dropout_rate=args.dropout_rate,
            ff_dim=args.ff_dim,
            normalize_before=args.normalize_before,
            norm_type=args.norm_type,
        )


if __name__ == "__main__":
    run_multi_split_experiment(TSMixerMultiSplitExperiment)
