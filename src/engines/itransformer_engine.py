from src.base.torch_engine import TorchEngine

import torch
from typing import Tuple, Dict


class iTransformer_Engine(TorchEngine):
    """
    iTransformer pytorch trainer implemented in the sklearn fashion.

    iTransformer uses an encoder-only architecture with inverted dimensions,
    so no decoder input creation is needed (unlike Autoformer/Informer).
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN normalization.

        iTransformer input shape (after _prepare_batch): [batch, seq_len, channels]
        RevIN expects: [batch, seq_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="norm")

    def _revin_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RevIN denormalization.

        iTransformer output shape: [batch, pred_len, channels]
        RevIN expects: [batch, pred_len, channels]
        No transpose needed.
        """
        if not self._use_revin or self._revin is None:
            return x
        return self._revin(x, mode="denorm")

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = super()._prepare_batch(batch)

        # iTransformer expects [batch, seq_len, channels]
        batch_dict["x"] = batch_dict["x"].permute(0, 2, 1).contiguous()
        batch_dict["y"] = batch_dict["y"].permute(0, 2, 1).contiguous()

        # x_mark and y_mark are already [batch, seq_len, n_features]
        return batch_dict

    def _prepare_ground_truths(self, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Prepare ground truths for iTransformer.

        iTransformer doesn't use label_len (encoder-only), so y_batch
        should be [batch, pred_len, channels]. We just return the last
        pred_len steps to be safe.

        Args:
            y_batch: Ground truth tensor [batch, pred_len, channels]

        Returns:
            Ground truth tensor [batch, pred_len, channels]
        """
        return y_batch[:, -self.model.pred_len :, :].contiguous()

    def _prepare_test_data(self, preds, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = preds.permute(0, 2, 1).contiguous()
        labels = labels.permute(0, 2, 1).contiguous()
        return preds, labels

    def _forward_pass(
        self,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        # Get batch data
        batch_dict = self._get_batch_inputs()

        x_enc = batch_dict["x"]
        x_mark_enc = batch_dict.get("x_mark")

        # Handle time features
        if x_mark_enc is None:
            # Create dummy time features if not provided
            batch_size = x_enc.shape[0]
            seq_len = x_enc.shape[1]
            x_mark_enc = torch.zeros(
                [batch_size, seq_len, 4],
                device=x_enc.device,
                dtype=x_enc.dtype,
            )

        # Forward pass - iTransformer is encoder-only
        # x_dec and x_mark_dec are passed as None for API compatibility
        out_batch = self.model(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=None,
            x_mark_dec=None,
            flatten_output=False,
        )

        return out_batch
