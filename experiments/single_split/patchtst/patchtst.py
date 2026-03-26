from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.formers.patchtst import PatchTST
from src.engines.patchtst_engine import PatchTST_Engine
from src.utils.args import get_patchtst_config
import torch
import numpy as np
import csv


class PatchTSTExperiment(TorchSingleSplitExperiment):
    """PatchTST-specific training implementation."""

    def get_config_parser(self):
        return get_patchtst_config()

    def get_model_name(self):
        return "patchtst"

    def get_engine_class(self):
        return PatchTST_Engine

    def get_metrics(self):
        return ["mse", "mae", "mape", "rmse"]

    def create_model(self, args, dataloader):
        # automatic retrieval of input_channels
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

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

    def get_revin_num_features(self, args):
        """PatchTST uses enc_in for number of channels."""
        return args.enc_in

    # quick fix for the hessian calc for patchtst
    def _run_hessian_analysis(
        self,
        args,
        model,
        dataloader,
        log_dir,
        logger,
        loss_fn,
    ):
        from third_party.utils.pyhessian.pyhessian import hessian
        from third_party.utils.pyhessian.density_plot import get_esd_plot

        logger.info("Computing Hessian analysis...")

        # wrapper class that adapts pathctst for pyhessian
        class PatchTSTHessianWrapper(torch.nn.Module):
            def __init__(self, patchtst_model):
                super().__init__()
                self.patchtst_model = patchtst_model

            def forward(self, x):
                # Dataloader provides: [batch, channels, seq_len]
                # PatchTST expects:    [batch, seq_len, channels]
                x = x.permute(0, 2, 1)

                # PatchTST outputs: [batch, pred_len, channels]
                out = self.patchtst_model(x)

                # return as [batch, channels, pred_len] to match target format
                return out.permute(0, 2, 1)

        wrapped_model = PatchTSTHessianWrapper(model)

        hessian_comp = hessian(
            wrapped_model, loss_fn, dataloader=dataloader["train_loader"], cuda=args.device
        )

        trace_estimate = np.mean(hessian_comp.trace())
        num_params = sum(p.numel() for p in model.parameters())
        average_curvature = trace_estimate / num_params

        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1)
        max_eigenvalue = top_eigenvalues[0]

        print(f"Max eigenvalue: {max_eigenvalue}")
        print(f"Trace: {trace_estimate}")
        print(f"Parameters: {num_params}")
        print(f"Average curvature: {average_curvature}")

        save_dir = log_dir / "statistics" / "hessian_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / f"hessian_metrics_s{args.seed}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["max_eigenvalue", "trace", "parameters", "average_curvature"])
            writer.writerow([max_eigenvalue, trace_estimate, num_params, average_curvature])

        logger.info(f"Hessian metrics saved to: {csv_path}")

        # density_eigen, density_weight = hessian_comp.density()
        # get_esd_plot(density_eigen, density_weight, log_dir)

    def post_training_hooks(self, args, model, dataloader, log_dir, logger, loss_fn):
        """Run Hessian analysis after training if enabled."""
        # Call parent's Hessian analysis if enabled
        super().post_training_hooks(args, model, dataloader, log_dir, logger, loss_fn)


if __name__ == "__main__":
    run_single_split_experiment(PatchTSTExperiment)
