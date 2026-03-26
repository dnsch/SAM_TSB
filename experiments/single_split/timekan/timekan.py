import torch
import numpy as np
import csv

from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.timekan.models.TimeKAN import TimeKAN
from src.engines.timekan_engine import TimeKAN_Engine
from src.utils.args import get_timekan_config


class TimeKANExperiment(TorchSingleSplitExperiment):
    """TimeKAN-specific training implementation."""

    def get_config_parser(self):
        return get_timekan_config()

    def get_model_name(self):
        return "timekan"

    def get_engine_class(self):
        return TimeKAN_Engine

    def get_log_dir_params(self, args):
        """Get parameter-specific part of log directory path."""
        base_params = (
            f"seq_len_{args.seq_len}_pred_len_{args.pred_len}_"
            f"d_model_{args.d_model}_e_layers_{args.e_layers}_"
            f"bs_{args.batch_size}"
        )
        if getattr(args, "sam", False):
            return f"{base_params}_rho_{args.rho}"
        elif getattr(args, "gsam", False):
            return f"{base_params}_gsam_alpha_{args.gsam_alpha}_rho_max_{args.gsam_rho_max}"
        return base_params

    def get_metrics(self):
        """Override to specify TimeKAN metrics."""
        return ["mse", "mae", "mape", "rmse"]

    def get_dataloader_kwargs(self, args):
        """Override to add TimeKAN-specific dataloader parameters."""
        kwargs = super().get_dataloader_kwargs(args)
        kwargs["use_time_features"] = getattr(args, "use_time_features", False)
        kwargs["freq"] = getattr(args, "freq", "h")
        kwargs["embed"] = getattr(args, "embed", "timeF")
        return kwargs

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
    ):
        """Override to add TimeKAN-specific engine parameters."""
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
        )
        kwargs["use_time_features"] = getattr(args, "use_time_features", False)
        return kwargs

    def create_model(self, args, dataloader):
        """Create TimeKAN model instance."""
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        return TimeKAN(
            input_channels=self._input_channels,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            label_len=getattr(args, "label_len", 0),
            d_model=getattr(args, "d_model", 32),
            e_layers=getattr(args, "e_layers", 2),
            down_sampling_layers=getattr(args, "down_sampling_layers", 2),
            down_sampling_window=getattr(args, "down_sampling_window", 2),
            moving_avg=getattr(args, "moving_avg", 25),
            embed=getattr(args, "embed", "timeF"),
            freq=getattr(args, "freq", "h"),
            dropout=getattr(args, "dropout", 0.1),
            use_norm=getattr(args, "use_norm", 1),
            channel_independence=getattr(args, "channel_independence", 1),
            begin_order=getattr(args, "begin_order", 2),
            use_future_temporal_feature=getattr(args, "use_future_temporal_feature", False),
            c_out=getattr(args, "c_out", None),
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
        """Custom Hessian analysis for TimeKAN model."""
        from third_party.utils.pyhessian.pyhessian import hessian

        logger.info("Computing Hessian analysis...")

        # Wrapper for TimeKAN model
        # TimeKAN forward expects [B, C, L] and outputs [B, C, pred_len]
        # Dataloader provides [B, C, L] and targets [B, C, pred_len]
        # So no dimension transformation needed, but we need a clean wrapper
        class TimeKANHessianWrapper(torch.nn.Module):
            def __init__(self, timekan_model):
                super().__init__()
                self.timekan_model = timekan_model

            def forward(self, x):
                # TimeKAN expects [B, C, L], outputs [B, C, pred_len]
                # The model handles transpose internally
                out = self.timekan_model(x, flatten_output=False)
                return out

        # Dataloader wrapper to handle potential time features
        class SimpleDataloaderWrapper:
            def __init__(self, original_loader):
                self.original_loader = original_loader

            def __iter__(self):
                for batch in self.original_loader:
                    if len(batch) == 4:
                        # (x, y, x_mark, y_mark) -> (x, y)
                        x, y, _, _ = batch
                    elif len(batch) == 2:
                        x, y = batch
                    else:
                        # Handle unexpected batch format
                        x = batch[0]
                        y = batch[1]
                    yield x, y

            def __len__(self):
                return len(self.original_loader)

        wrapped_model = TimeKANHessianWrapper(model)
        wrapped_dataloader = SimpleDataloaderWrapper(dataloader["train_loader"])

        use_cuda = torch.cuda.is_available() and "cuda" in args.device

        try:
            hessian_comp = hessian(
                wrapped_model,
                loss_fn,
                dataloader=wrapped_dataloader,
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

        except Exception as e:
            logger.error(f"Hessian analysis failed: {e}")
            logger.info("Attempting alternative Hessian computation...")

            # Alternative: compute on a single batch
            self._run_single_batch_hessian(args, model, dataloader, log_dir, logger, loss_fn)

    def _run_single_batch_hessian(
        self,
        args,
        model,
        dataloader,
        log_dir,
        logger,
        loss_fn,
    ):
        """Fallback Hessian analysis using a single batch."""
        from third_party.utils.pyhessian.pyhessian import hessian

        logger.info("Running single-batch Hessian analysis...")

        # Get a single batch
        train_loader = dataloader["train_loader"]
        batch = next(iter(train_loader))

        if len(batch) == 4:
            x, y, _, _ = batch
        else:
            x, y = batch

        device = torch.device(
            args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
        )
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)

        use_cuda = device.type == "cuda"

        try:
            hessian_comp = hessian(
                model,
                loss_fn,
                data=(x, y),
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
            csv_path = save_dir / f"hessian_metrics_s{args.seed}_single_batch.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["max_eigenvalue", "trace", "parameters", "average_curvature"])
                writer.writerow([max_eigenvalue, trace_estimate, num_params, average_curvature])

            logger.info(f"Hessian metrics (single batch) saved to: {csv_path}")

        except Exception as e:
            logger.error(f"Single-batch Hessian analysis also failed: {e}")
            logger.error(
                "TimeKAN's ChebyKAN layers may have gradient issues incompatible with Hessian computation."
            )

    def post_training_hooks(self, args, model, dataloader, log_dir, logger, loss_fn):
        """Run post-training analysis if enabled."""
        if getattr(args, "hessian_analysis", False):
            self._run_hessian_analysis(args, model, dataloader, log_dir, logger, loss_fn)


if __name__ == "__main__":
    run_single_split_experiment(TimeKANExperiment)
