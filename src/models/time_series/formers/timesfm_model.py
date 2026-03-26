import torch
import numpy as np
import timesfm
from tqdm import tqdm
from src.base.model import BaseModel

torch.set_float32_matmul_precision("high")


class TimesFM(BaseModel):
    """
    TimesFM 2.5: Google's Time Series Foundation Model for zero-shot forecasting.
    see: https://github.com/google-research/timesfm/blob/master/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py
    """

    def __init__(
        self,
        seq_len=512,
        pred_len=96,
        input_channels=None,
        model_name="google/timesfm-2.5-200m-pytorch",
        backend="gpu",
        batch_size=256,
        normalize_inputs=False,  # TimesFM's internal normalization
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len, input_channels=input_channels)

        self.batch_size = batch_size
        self.device = "cuda" if (backend == "gpu" and torch.cuda.is_available()) else "cpu"

        # Load model
        self.tfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_name,
            device=self.device,
        )

        # Compile with forecast config
        self.tfm_model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=normalize_inputs,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def forward(self, x, flatten_output=False):
        B, C, L = x.shape
        device = x.device
        dtype = x.dtype

        # Flatten: (B, C, L) -> (B*C, L)
        x_flat = x.reshape(B * C, L)
        total_series = x_flat.shape[0]

        # Convert to numpy
        x_np = x_flat.cpu().numpy()

        # Batched inference with progress bar
        all_predictions = []

        with torch.no_grad():
            for i in tqdm(
                range(0, total_series, self.batch_size), desc="TimesFM Forecasting", leave=False
            ):
                batch = x_np[i : i + self.batch_size]
                inputs_list = [batch[j] for j in range(batch.shape[0])]

                point_forecast, _ = self.tfm_model.forecast(
                    horizon=self.pred_len,
                    inputs=inputs_list,
                )
                all_predictions.append(point_forecast)

        # Concatenate predictions
        predictions = np.concatenate(all_predictions, axis=0)

        # Handle length mismatch (safety check)
        if predictions.shape[1] > self.pred_len:
            predictions = predictions[:, : self.pred_len]
        elif predictions.shape[1] < self.pred_len:
            pad_len = self.pred_len - predictions.shape[1]
            predictions = np.pad(predictions, ((0, 0), (0, pad_len)), mode="edge")

        # Convert back to tensor: (B*C, pred_len) -> (B, C, pred_len)
        output = torch.tensor(predictions, dtype=dtype, device=device)
        output = output.reshape(B, C, self.pred_len)

        if flatten_output:
            output = output.reshape(B, -1)

        return output

    def param_num(self):
        # see: https://github.com/google-research/timesfm
        return 200_000_000
