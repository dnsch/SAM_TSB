from typing import Dict, Any
import argparse
import torch
import numpy as np
import csv

from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.formers.autoformer import Autoformer
from src.engines.autoformer_engine import Autoformer_Engine
from src.utils.args import get_autoformer_config


class AutoformerStandardExperiment(TorchSingleSplitExperiment):
    """Autoformer-specific training implementation."""

    def __init__(self):
        super().__init__()
        self._dec_in = None
        self._c_out = None

    def get_config_parser(self):
        return get_autoformer_config()

    def get_model_name(self):
        return "autoformer"

    def get_engine_class(self):
        return Autoformer_Engine

    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for dataloader initialization with time features for Autoformer."""
        kwargs = super().get_dataloader_kwargs(args)

        kwargs.update(
            {
                "use_time_features": getattr(args, "use_time_features", True),
                "freq": getattr(args, "freq", "h"),
                "timeenc": 1 if getattr(args, "embed", "timeF") == "timeF" else 0,
                "embed": getattr(args, "embed", "timeF"),
                "label_len": getattr(args, "label_len", 48),
            }
        )

        return kwargs

    def create_model(self, args, dataloader):
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )
        self._dec_in = self.dataloader_instance.get_input_channels(getattr(args, "dec_in", None))
        self._c_out = self.dataloader_instance.get_input_channels(getattr(args, "c_out", None))

        return Autoformer(
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=self._input_channels,
            dec_in=self._dec_in,
            c_out=self._c_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            moving_avg=args.moving_avg,
            factor=args.factor,
            dropout=args.dropout,
            embed_type=args.embed_type,
            embed=args.embed,
            freq=args.freq,
            activation=args.activation,
            output_attention=args.output_attention,
        )

    def _run_hessian_analysis(
        self,
        args,
        model,
        dataloader,
        log_dir,
        logger,
        loss_fn,
    ):
        """Custom Hessian analysis for Autoformer that handles time features."""
        from third_party.utils.pyhessian.pyhessian import hessian
        from third_party.utils.pyhessian.density_plot import get_esd_plot

        logger.info("Computing Hessian analysis...")

        # Wrapper class that adapts Autoformer for pyhessian
        # pyhessian expects: model(inputs) -> outputs, with dataloader yielding (inputs, targets)
        class AutoformerHessianWrapper(torch.nn.Module):
            def __init__(self, autoformer_model, label_len, pred_len):
                super().__init__()
                self.autoformer_model = autoformer_model
                self.label_len = label_len
                self.pred_len = pred_len

            def _create_decoder_input(self, x_enc):
                """Create decoder input from encoder input."""
                batch_size = x_enc.shape[0]
                channels = x_enc.shape[2]

                # Get last label_len steps from encoder input
                dec_input = x_enc[:, -self.label_len :, :].clone()

                # Append pred_len zeros
                zeros = torch.zeros(
                    [batch_size, self.pred_len, channels],
                    device=x_enc.device,
                    dtype=x_enc.dtype,
                )
                x_dec = torch.cat([dec_input, zeros], dim=1)
                return x_dec

            def forward(self, x):
                # Dataloader provides: [batch, channels, seq_len]
                # Autoformer expects:  [batch, seq_len, channels]
                x_enc = x.permute(0, 2, 1)

                batch_size = x_enc.shape[0]
                seq_len = x_enc.shape[1]

                # Create decoder input
                x_dec = self._create_decoder_input(x_enc)

                # Create dummy time features (zeros) since we only have x, y from wrapper
                x_mark_enc = torch.zeros(
                    [batch_size, seq_len, 4],
                    device=x_enc.device,
                    dtype=x_enc.dtype,
                )
                x_mark_dec = torch.zeros(
                    [batch_size, self.label_len + self.pred_len, 4],
                    device=x_enc.device,
                    dtype=x_enc.dtype,
                )

                # Forward pass
                out = self.autoformer_model(
                    x_enc=x_enc,
                    x_mark_enc=x_mark_enc,
                    x_dec=x_dec,
                    x_mark_dec=x_mark_dec,
                    enc_self_mask=None,
                    dec_self_mask=None,
                    dec_enc_mask=None,
                    flatten_output=False,
                )

                # Autoformer outputs: [batch, pred_len, channels]
                # Return as [batch, channels, pred_len] to match target format
                return out.permute(0, 2, 1)

        # Create a dataloader wrapper that only yields (x, y) pairs
        class SimpleDataloaderWrapper:
            def __init__(self, original_loader, label_len, pred_len):
                self.original_loader = original_loader
                self.label_len = label_len
                self.pred_len = pred_len

            def __iter__(self):
                for batch in self.original_loader:
                    if len(batch) == 4:
                        # (x, y, x_mark, y_mark) -> (x, y)
                        x, y, _, _ = batch
                    else:
                        # (x, y) -> (x, y)
                        x, y = batch

                    # y from dataloader has shape [batch, channels, label_len + pred_len]
                    # We need only the last pred_len for target
                    y_target = y[:, :, -self.pred_len :]

                    yield x, y_target

            def __len__(self):
                return len(self.original_loader)

        wrapped_model = AutoformerHessianWrapper(
            model,
            label_len=args.label_len,
            pred_len=args.pred_len,
        )

        wrapped_dataloader = SimpleDataloaderWrapper(
            dataloader["train_loader"],
            label_len=args.label_len,
            pred_len=args.pred_len,
        )

        hessian_comp = hessian(
            wrapped_model,
            loss_fn,
            dataloader=wrapped_dataloader,
            cuda=args.device,
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

    def post_training_hooks(self, args, model, dataloader, log_dir, logger, loss_fn):
        """Run Hessian analysis after training if enabled."""
        if getattr(args, "hessian_analysis", False):
            self._run_hessian_analysis(args, model, dataloader, log_dir, logger, loss_fn)


if __name__ == "__main__":
    run_single_split_experiment(AutoformerStandardExperiment)
