from src.base.torch_multi_split_experiment import (
    TorchMultiSplitExperiment,
    run_multi_split_experiment,
)
from src.models.time_series.samformer import SAMFormer
from src.engines.samformer_engine import SAMFormer_Engine
from src.utils.args import get_samformer_multi_split_experiment_config


class SAMFormerMultiSplitExperiment(TorchMultiSplitExperiment):
    """SAMFormer multi split experiment implementation."""

    def get_config_parser(self):
        return get_samformer_multi_split_experiment_config()

    def get_model_name(self):
        return "samformer"

    def get_engine_class(self):
        return SAMFormer_Engine

    # def get_log_dir_suffix(self, args):
    #     """Override to use 'simple_transformer' when SAM is disabled."""
    #     if getattr(args, "sam", False):
    #         return "samformer"
    #     elif getattr(args, "gsam", False):
    #         return "samformerGSAM"
    #     return "simple_transformer"

    def get_metrics(self):
        """Override to specify SAMFormer metrics."""
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
        """Override to add SAMFormer-specific engine parameters."""
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
        # Only plot attention for first split to avoid clutter
        kwargs["plot_attention"] = getattr(args, "plot_attention", True) and split_idx == 0
        return kwargs

    def create_model(self, args, dataloader):
        """Create SAMFormer model instance."""
        return SAMFormer(
            input_channels=self._input_channels,
            seq_len=args.seq_len,
            hid_dim=args.hid_dim,
            pred_len=args.pred_len,
            plot_attention=getattr(args, "plot_attention", True),
        )


if __name__ == "__main__":
    run_multi_split_experiment(SAMFormerMultiSplitExperiment)
