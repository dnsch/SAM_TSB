import torch
import numpy as np
import csv

from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.formers.nlinear import NLinear
from src.engines.nlinear_engine import NLinear_Engine
from src.utils.args import get_nlinear_config


class NLinearStandardExperiment(TorchSingleSplitExperiment):
    """NLinear-specific training implementation."""

    def get_config_parser(self):
        return get_nlinear_config()

    def get_model_name(self):
        return "nlinear"

    def get_engine_class(self):
        return NLinear_Engine

    def create_model(self, args, dataloader):
        # automatic retrieval of input_channels
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        return NLinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=self._input_channels,
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
        """Custom Hessian analysis for NLinear model."""
        from third_party.utils.pyhessian.pyhessian import hessian

        logger.info("Computing Hessian analysis...")

        # Wrapper for NLinear model to handle dimension transformation
        class NLinearHessianWrapper(torch.nn.Module):
            def __init__(self, nlinear_model):
                super().__init__()
                self.nlinear_model = nlinear_model

            def forward(self, x):
                # Dataloader provides: [batch, channels, seq_len]
                # NLinear expects: [batch, seq_len, channels]
                x = x.permute(0, 2, 1)

                # NLinear outputs: [batch, pred_len, channels]
                out = self.nlinear_model(x)

                # Return as [batch, channels, pred_len] to match target format
                return out.permute(0, 2, 1)

        wrapped_model = NLinearHessianWrapper(model)

        use_cuda = torch.cuda.is_available() and "cuda" in args.device

        hessian_comp = hessian(
            wrapped_model,
            loss_fn,
            dataloader=dataloader["train_loader"],
            cuda=use_cuda,
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
        """Run Hessian analysis if enabled."""
        if getattr(args, "hessian_analysis", False):
            self._run_hessian_analysis(args, model, dataloader, log_dir, logger, loss_fn)


if __name__ == "__main__":
    run_single_split_experiment(NLinearStandardExperiment)
