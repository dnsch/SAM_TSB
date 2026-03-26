from src.base.torch_engine import TorchEngine
import torch


class TimeKAN_Engine(TorchEngine):
    """
    TimeKAN PyTorch trainer.

    Handles the specific input format requirements for TimeKAN.
    """

    def __init__(
        self,
        use_time_features=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_time_features = use_time_features

    def _forward_pass(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass through TimeKAN model.

        Args:
            x_batch: Input batch tensor of shape [B, C, L]

        Returns:
            Model output tensor of shape [B, C, pred_len]
        """
        batch_inputs = self._get_batch_inputs()

        # Get time features if available and enabled
        x_mark = batch_inputs.get("x_mark") if self.use_time_features else None

        # Forward pass
        out = self.model(
            x=x_batch,
            flatten_output=False,
            x_mark_enc=x_mark,
        )

        return out
