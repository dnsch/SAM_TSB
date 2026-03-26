from src.base.torch_multi_split_experiment import (
    TorchMultiSplitExperiment,
    run_multi_split_experiment,
)
from src.models.time_series.formers.patchtst import PatchTST
from src.engines.patchtst_engine import PatchTST_Engine
from src.utils.args import get_patchtst_multi_split_experiment_config


class PatchTSTMultiSplitExperiment(TorchMultiSplitExperiment):
    """PatchTST multi split experiment implementation."""

    def get_config_parser(self):
        return get_patchtst_multi_split_experiment_config()

    def get_model_name(self):
        return "patchtst"

    def get_engine_class(self):
        return PatchTST_Engine

    # changed save path creation
    # def get_log_dir_suffix(self, args):
    #     """Override to handle SAM/GSAM suffixes."""
    #     if getattr(args, "sam", False):
    #         return "patchtstSAM"
    #     elif getattr(args, "gsam", False):
    #         return "patchtstGSAM"
    #     return "patchtst"

    def get_metrics(self):
        """Override to specify PatchTST metrics."""
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
        """Override to add PatchTST-specific engine parameters."""
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
        """Create PatchTST model instance."""
        return PatchTST(
            enc_in=self._input_channels,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            e_layers=args.e_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,
            patch_len=args.patch_len,
            stride=args.stride,
            padding_patch=args.padding_patch,
            decomposition=args.decomposition,
            kernel_size=args.kernel_size,
            individual=args.individual,
            max_seq_len=args.seq_len,
            d_k=args.d_k,
            d_v=args.d_v,
            norm=args.norm,
            attn_dropout=args.attn_dropout,
            act=args.activation,
            key_padding_mask=args.key_padding_mask,
            padding_var=args.padding_var,
            attn_mask=args.attn_mask,
            res_attention=args.res_attention,
            pre_norm=args.pre_norm,
            store_attn=args.store_attn,
            pe=args.pe,
            learn_pe=args.learn_pe,
            pretrain_head=args.pretrain_head,
            head_type=args.head_type,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    run_multi_split_experiment(PatchTSTMultiSplitExperiment)
