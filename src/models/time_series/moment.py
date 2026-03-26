import torch
from momentfm import MOMENTPipeline

from src.base.model import BaseModel


class MOMENT(BaseModel):
    """
    Wrapper for the MOMENT time series foundation model for forecasting.

    Input preparation (padding to 512) is handled by the engine.
    """

    MOMENT_CONTEXT_LENGTH = 512

    def __init__(
        self,
        seq_len=512,
        pred_len=96,
        input_channels=None,
        model_name="AutonLab/MOMENT-1-large",
        head_dropout=0.1,
        weight_decay=0,
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
        **kwargs,
    ):
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            input_channels=input_channels,
        )

        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.freeze_embedder = freeze_embedder
        self.freeze_head = freeze_head

        # Load MOMENT model
        self.moment = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": pred_len,
                "head_dropout": head_dropout,
                "weight_decay": weight_decay,
                "freeze_encoder": freeze_encoder,
                "freeze_embedder": freeze_embedder,
                "freeze_head": freeze_head,
            },
        )

        # Initialize the model (required by MOMENT)
        self.moment.init()

    def forward(self, x, input_mask=None, flatten_output=True):
        """
        Forward pass through MOMENT.

        Args:
            x: Input tensor of shape (batch_size, n_channels, 512)
            input_mask: Optional mask of shape (batch_size, 512).
                        1 for observed timesteps, 0 for padding.
            flatten_output: If True, return shape (batch_size, n_channels * pred_len)
                            If False, return shape (batch_size, n_channels, pred_len)

        Returns:
            Forecast tensor
        """
        # Forward pass through MOMENT
        output = self.moment(x_enc=x, input_mask=input_mask)

        # output.forecast shape: (batch_size, n_channels, pred_len)
        out = output.forecast

        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])
        else:
            return out

    def get_unfrozen_parameters(self):
        """Return a list of unfrozen parameter names (useful for debugging)."""
        return [name for name, param in self.moment.named_parameters() if param.requires_grad]
