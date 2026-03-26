import torch
import torch.nn as nn

from src.base.model import BaseModel


class NLinear(BaseModel):
    """
    taken from
    https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/models/NLinear.py

    Normalization-Linear
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len)

        self.channels = enc_in
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
