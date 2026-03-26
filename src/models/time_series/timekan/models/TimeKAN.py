import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.base.model import BaseModel
from src.models.time_series.timekan.layers.Autoformer_EncDec import series_decomp
from src.models.time_series.timekan.layers.Embed import DataEmbedding_wo_pos
from src.models.time_series.timekan.layers.StandardNorm import Normalize
from src.models.time_series.timekan.layers.ChebyKANLayer import ChebyKANLinear


"""
Taken from:https://github.com/huangst21/TimeKAN/blob/main/models/TimeKAN.py
"""


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(in_features, out_features, order)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x


class FrequencyDecomp(nn.Module):
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(FrequencyDecomp, self).__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i)),
                self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i - 1)),
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros(
            [x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype
        ).to(x_fft.device)
        out_fft[:, :, : seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class FrequencyMixing(nn.Module):
    def __init__(self, seq_len, d_model, down_sampling_window, down_sampling_layers, begin_order):
        super(FrequencyMixing, self).__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.front_block = M_KAN(
            d_model,
            self.seq_len // (self.down_sampling_window**self.down_sampling_layers),
            order=begin_order,
        )

        self.front_blocks = torch.nn.ModuleList(
            [
                M_KAN(
                    d_model,
                    self.seq_len
                    // (self.down_sampling_window ** (self.down_sampling_layers - i - 1)),
                    order=i + begin_order + 1,
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i)),
                self.seq_len // (self.down_sampling_window ** (self.down_sampling_layers - i - 1)),
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros(
            [x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype
        ).to(x_fft.device)
        out_fft[:, :, : seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class M_KAN(nn.Module):
    def __init__(self, d_model, seq_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(ChebyKANLayer(d_model, d_model, order))
        self.conv = BasicConv(d_model, d_model, kernel_size=3, degree=order, groups=d_model)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1 + x2
        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        degree,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        act=False,
        bn=False,
        bias=False,
        dropout=0.0,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TimeKAN(BaseModel):
    """
    TimeKAN: Kolmogorov-Arnold Networks for Time Series Forecasting.

    Combines frequency decomposition with multi-scale processing using
    Chebyshev polynomial-based KAN layers.

    Input shape: [B, C, L] (batch, channels, seq_len)
    Output shape: [B, C, pred_len] or [B, C*pred_len] if flatten_output=True
    """

    def __init__(
        self,
        input_channels,
        seq_len,
        pred_len,
        label_len=0,
        d_model=32,
        e_layers=2,
        down_sampling_layers=2,
        down_sampling_window=2,
        moving_avg=25,
        embed="timeF",
        freq="h",
        dropout=0.1,
        use_norm=1,
        channel_independence=1,
        begin_order=2,
        use_future_temporal_feature=False,
        c_out=None,
        task_name="long_term_forecast",
        **kwargs,
    ):
        """
        Initialize TimeKAN model.

        Args:
            input_channels: Number of input channels (enc_in)
            seq_len: Input sequence length
            pred_len: Prediction horizon length
            label_len: Decoder start token length (default: 0)
            d_model: Model dimension (default: 32)
            e_layers: Number of encoder layers (default: 2)
            down_sampling_layers: Number of downsampling layers (default: 2)
            down_sampling_window: Downsampling window size (default: 2)
            moving_avg: Moving average kernel size for decomposition (default: 25)
            embed: Time feature embedding type (default: 'timeF')
            freq: Frequency for time features (default: 'h')
            dropout: Dropout rate (default: 0.1)
            use_norm: Whether to use normalization, 0 or 1 (default: 1)
            channel_independence: Channel independence flag, 0 or 1 (default: 1)
            begin_order: Beginning order for Chebyshev polynomials (default: 2)
            use_future_temporal_feature: Use future temporal features (default: False)
            c_out: Number of output channels (default: same as input_channels)
            task_name: Task name (default: 'long_term_forecast')
        """
        super().__init__(
            input_channels=input_channels,
            seq_len=seq_len,
            pred_len=pred_len,
        )

        # Store config values
        self.task_name = task_name
        self.label_len = label_len
        self.d_model = d_model
        self.e_layers = e_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.channel_independence = channel_independence
        self.use_future_temporal_feature = use_future_temporal_feature
        self.use_norm = use_norm
        self.enc_in = input_channels
        self.c_out = c_out if c_out is not None else input_channels

        # Validate seq_len is compatible with downsampling
        min_seq_len = down_sampling_window**down_sampling_layers
        if seq_len < min_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) must be >= {min_seq_len} "
                f"(down_sampling_window^down_sampling_layers = "
                f"{down_sampling_window}^{down_sampling_layers})"
            )

        # Build model components
        self.res_blocks = nn.ModuleList(
            [
                FrequencyDecomp(seq_len, down_sampling_window, down_sampling_layers)
                for _ in range(e_layers)
            ]
        )
        self.add_blocks = nn.ModuleList(
            [
                FrequencyMixing(
                    seq_len, d_model, down_sampling_window, down_sampling_layers, begin_order
                )
                for _ in range(e_layers)
            ]
        )

        self.preprocess = series_decomp(moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(
                    input_channels,
                    affine=True,
                    non_norm=True if use_norm == 0 else False,
                )
                for _ in range(down_sampling_layers + 1)
            ]
        )

        self.projection_layer = nn.Linear(d_model, 1, bias=True)
        self.predict_layer = nn.Linear(seq_len, pred_len)

        # Initialize weights
        self._init_weights()

    def _multi_level_process_inputs(self, x_enc):
        """
        Process inputs at multiple scales using average pooling.

        Args:
            x_enc: Input tensor of shape [B, L, C]

        Returns:
            List of tensors at different scales
        """
        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        # B,L,C -> B,C,L
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        return x_enc_sampling_list

    def forecast(self, x_enc):
        """
        Forecasting forward pass.

        Args:
            x_enc: Input tensor of shape [B, L, C]

        Returns:
            Output tensor of shape [B, pred_len, c_out]
        """
        x_enc = self._multi_level_process_inputs(x_enc)
        x_list = []

        for i, x in enumerate(x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, "norm")
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in enumerate(x_list):
            enc_out = self.enc_embedding(x, None)  # [B*N, T, d_model]
            enc_out_list.append(enc_out)

        for i in range(self.e_layers):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        # Get batch size from first level input
        B = x_enc[0].size(0)

        dec_out = (
            self.projection_layer(dec_out)
            .reshape(B, self.c_out, self.pred_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def forward(self, x, flatten_output=False, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forward pass through TimeKAN.

        Args:
            x: Input tensor of shape [B, C, L] (batch, channels, seq_len)
            flatten_output: Whether to flatten output to [B, C*pred_len]
            x_mark_enc: Optional time features for encoder (not used in current implementation)
            x_dec: Optional decoder input (not used in forecasting)
            x_mark_dec: Optional time features for decoder (not used in forecasting)

        Returns:
            Output tensor of shape [B, C, pred_len] or [B, C*pred_len] if flattened
        """
        # Input x shape from dataloader: [B, C, L]
        # TimeKAN internal processing expects: [B, L, C]
        x_enc = x.transpose(1, 2)

        if self.task_name == "long_term_forecast":
            # Forecast output shape: [B, pred_len, c_out]
            dec_out = self.forecast(x_enc)
            # Convert to [B, C, pred_len] to match repository convention
            out = dec_out.transpose(1, 2)
        else:
            raise ValueError(f"Task '{self.task_name}' not implemented")

        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])

        return out
