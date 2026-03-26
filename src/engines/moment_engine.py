import sys
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.base.torch_engine import TorchEngine


class MOMENT_Engine(TorchEngine):
    """
    MOMENT PyTorch trainer with input preparation for the 512 context length
    and detailed progress tracking.
    """

    MOMENT_CONTEXT_LENGTH = 512

    def __init__(self, log_interval=10, **kwargs):
        """
        Args:
            log_interval: Log metrics every N batches (default: 10)
            **kwargs: Arguments passed to TorchEngine
        """
        super().__init__(**kwargs)

        self.log_interval = log_interval

        # Cache for input masks (avoid recreating every batch)
        self._cached_input_mask = None
        self._cached_mask_shape = None

    def _prepare_moment_input(self, x: torch.Tensor):
        """
        Prepare input for MOMENT by padding/truncating to 512 timesteps.

        Args:
            x: Input tensor of shape (batch_size, n_channels, seq_len)

        Returns:
            x_prepared: Tensor of shape (batch_size, n_channels, 512)
            input_mask: Tensor of shape (batch_size, 512) or None
        """
        batch_size, n_channels, input_len = x.shape

        if input_len == self.MOMENT_CONTEXT_LENGTH:
            return x, None

        elif input_len < self.MOMENT_CONTEXT_LENGTH:
            # Left-pad with zeros to keep recent values at the end
            pad_len = self.MOMENT_CONTEXT_LENGTH - input_len
            padding = torch.zeros(
                batch_size,
                n_channels,
                pad_len,
                device=x.device,
                dtype=x.dtype,
            )
            x_prepared = torch.cat([padding, x], dim=-1)

            # Create or reuse input mask (0 for padded, 1 for actual data)
            mask_shape = (batch_size, self.MOMENT_CONTEXT_LENGTH)
            if self._cached_input_mask is None or self._cached_mask_shape != mask_shape:
                input_mask = torch.zeros(batch_size, self.MOMENT_CONTEXT_LENGTH, device=x.device)
                input_mask[:, -input_len:] = 1
                self._cached_input_mask = input_mask
                self._cached_mask_shape = mask_shape
            else:
                input_mask = self._cached_input_mask.to(x.device)

            return x_prepared, input_mask

        else:  # input_len > MOMENT_CONTEXT_LENGTH
            # Truncate: keep most recent values
            x_prepared = x[:, :, -self.MOMENT_CONTEXT_LENGTH :]
            return x_prepared, None

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass through MOMENT with input preparation.

        Args:
            x_batch: Input batch tensor of shape (batch_size, n_channels, seq_len)

        Returns:
            Model output tensor
        """
        # Prepare input (pad/truncate to 512)
        x_prepared, input_mask = self._prepare_moment_input(x_batch)

        # Forward pass through MOMENT
        # flatten_output=False because we handle reshaping in _prepare_predictions
        out_batch = self.model(x_prepared, input_mask=input_mask, flatten_output=False)

        return out_batch

    def train_batch(self):
        """
        Train for one epoch with detailed progress tracking.

        Returns:
            Dictionary of metric_name: mean_value pairs
        """
        self.model.train()
        batch_metrics = defaultdict(list)

        train_loader = self._get_train_loader()
        total_batches = len(train_loader)

        # Create progress bar with settings to prevent new lines
        pbar = tqdm(
            enumerate(train_loader),
            total=total_batches,
            desc="Training",
            file=sys.stdout,
            dynamic_ncols=True,
            position=0,
            leave=True,
            mininterval=0.5,  # Update at most every 0.5 seconds
        )

        running_loss = 0.0
        running_count = 0

        for batch_idx, batch in pbar:
            # Prepare data
            batch_dict = self._prepare_batch(batch)
            self._current_batch = batch_dict

            batch_dict = self._to_device(batch_dict)
            self._current_batch = batch_dict

            x_batch = batch_dict["x"]
            y_batch = batch_dict["y"]

            # RevIN Normalization
            x_batch_norm = self._revin_norm(x_batch)

            # Forward pass
            pred = self._forward_pass(x_batch_norm).contiguous()

            # Hook that allows to capture model internals
            self._on_forward_pass()

            # RevIN Denormalization
            pred = self._revin_denorm(pred)

            # Prepare predictions (put them in the right shape)
            prepared_pred = self._prepare_predictions(pred)
            prepared_y_batch = self._prepare_ground_truths(y_batch)

            # Compute loss
            loss = self._compute_loss(prepared_pred, prepared_y_batch)

            # Optimizer step
            self._optimizer_step(loss, x_batch, y_batch)

            # Compute all metrics
            metrics = self._compute_metrics(prepared_pred, prepared_y_batch)

            # Store all metrics
            for metric_name, metric_value in metrics.items():
                batch_metrics[metric_name].append(metric_value)

            # Update running statistics
            running_loss += metrics.get(self._get_loss_name(), 0.0)
            running_count += 1

            # Update progress bar (use set_postfix_str for cleaner output)
            avg_loss = running_loss / running_count
            current_loss = metrics.get(self._get_loss_name(), 0.0)
            pbar.set_postfix_str(f"loss={current_loss:.4f}, avg={avg_loss:.4f}")

        # Ensure final state is printed
        pbar.close()
        sys.stdout.flush()

        # Return mean of all metrics
        return {name: np.mean(values) for name, values in batch_metrics.items()}

    def evaluate(self, mode: str):
        """
        Evaluate the model with progress tracking.

        Args:
            mode: 'val' or 'test'

        Returns:
            Dictionary of metric_name: value pairs
        """
        if mode == "test":
            self.load_model(self._save_path)

        self.model.eval()

        preds, labels = [], []

        loader_key = f"{mode}_loader"
        loader = self._get_dataloader(loader_key)

        # Create progress bar for evaluation
        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Eval ({mode})",
            file=sys.stdout,
            dynamic_ncols=True,
            position=0,
            leave=True,
            mininterval=0.5,
        )

        with torch.no_grad():
            for batch in pbar:
                batch_dict = self._prepare_batch(batch)
                self._current_batch = batch_dict

                # Move to device
                batch_dict = self._to_device(batch_dict)
                self._current_batch = batch_dict

                x_batch = batch_dict["x"]
                y_batch = batch_dict["y"]

                # RevIN Normalization
                x_batch_norm = self._revin_norm(x_batch)

                pred = self._forward_pass(x_batch_norm)

                # RevIN Denormalization
                pred = self._revin_denorm(pred)

                preds.append(pred.squeeze(-1).cpu())
                labels.append(y_batch.squeeze(-1).cpu())

        pbar.close()
        sys.stdout.flush()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        prepared_preds = self._prepare_predictions(preds)
        prepared_labels = self._prepare_ground_truths(labels)

        if mode == "val":
            return self._compute_validation_metrics(prepared_preds, prepared_labels)
        elif mode == "test":
            return self._compute_test_metrics(prepared_preds, prepared_labels)
