from typing import Dict, Any
import argparse
import torch
import numpy as np
import csv

from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.formers.itransformer import iTransformer
from src.engines.itransformer_engine import iTransformer_Engine
from src.utils.args import get_itransformer_config


class iTransformerStandardExperiment(TorchSingleSplitExperiment):
    """iTransformer-specific training implementation."""

    def __init__(self):
        super().__init__()
        self._c_out = None

    def get_config_parser(self):
        return get_itransformer_config()

    def get_model_name(self):
        return "itransformer"

    def get_engine_class(self):
        return iTransformer_Engine

    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for dataloader initialization with time features for iTransformer."""
        kwargs = super().get_dataloader_kwargs(args)

        # Add time feature parameters for iTransformer
        # Note: iTransformer doesn't use label_len (encoder-only)
        kwargs.update(
            {
                "use_time_features": getattr(args, "use_time_features", True),
                "freq": getattr(args, "freq", "h"),
                "timeenc": 1 if getattr(args, "embed", "timeF") == "timeF" else 0,
                "embed": getattr(args, "embed", "timeF"),
                "label_len": 0,  # iTransformer is encoder-only, no label_len needed
            }
        )

        return kwargs

    def create_model(self, args, dataloader):
        # automatic retrieval of input_channels, and c_out if not set manually
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )
        self._c_out = self.dataloader_instance.get_input_channels(getattr(args, "c_out", None))

        return iTransformer(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=self._input_channels,
            c_out=self._c_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            factor=args.factor,
            dropout=args.dropout,
            embed=args.embed,
            freq=args.freq,
            activation=args.activation,
            use_norm=args.use_norm,
            output_attention=args.output_attention,
            class_strategy=args.class_strategy,
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
        """Custom Hessian analysis for iTransformer that handles time features."""
        from third_party.utils.pyhessian.pyhessian import hessian
        from third_party.utils.pyhessian.density_plot import get_esd_plot

        logger.info("Computing Hessian analysis...")

        # Wrapper class that adapts iTransformer for pyhessian
        class iTransformerHessianWrapper(torch.nn.Module):
            def __init__(self, itransformer_model, seq_len):
                super().__init__()
                self.itransformer_model = itransformer_model
                self.seq_len = seq_len

            def forward(self, x):
                # Dataloader provides: [batch, channels, seq_len]
                # iTransformer expects: [batch, seq_len, channels]
                x_enc = x.permute(0, 2, 1)

                batch_size = x_enc.shape[0]

                # Create dummy time features (zeros) since pyhessian only passes x
                x_mark_enc = torch.zeros(
                    [batch_size, self.seq_len, 4],
                    device=x_enc.device,
                    dtype=x_enc.dtype,
                )

                # Forward pass (encoder-only, no decoder inputs needed)
                out = self.itransformer_model(
                    x_enc=x_enc,
                    x_mark_enc=x_mark_enc,
                    flatten_output=False,
                )

                # iTransformer outputs: [batch, pred_len, channels]
                # Return as [batch, channels, pred_len] to match target format
                return out.permute(0, 2, 1)

        # Dataloader wrapper that strips time features
        class SimpleDataloaderWrapper:
            def __init__(self, original_loader):
                self.original_loader = original_loader

            def __iter__(self):
                for batch in self.original_loader:
                    if len(batch) == 4:
                        # (x, y, x_mark, y_mark) -> (x, y)
                        x, y, _, _ = batch
                    else:
                        # (x, y) -> (x, y)
                        x, y = batch

                    yield x, y

            def __len__(self):
                return len(self.original_loader)

        wrapped_model = iTransformerHessianWrapper(
            model,
            seq_len=args.seq_len,
        )

        wrapped_dataloader = SimpleDataloaderWrapper(dataloader["train_loader"])

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
    run_single_split_experiment(iTransformerStandardExperiment)
