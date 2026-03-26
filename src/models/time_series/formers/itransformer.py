import torch
import torch.nn as nn

from src.base.model import BaseModel
from src.models.time_series.formers.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.time_series.formers.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)
from src.models.time_series.formers.layers.Embed import DataEmbedding_inverted


class iTransformer(BaseModel):
    """
    iTransformer model
    see: https://github.com/thuml/iTransformer

    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_ff=2048,
        factor=5,
        dropout=0.05,
        embed="timeF",
        freq="h",
        activation="gelu",
        use_norm=True,
        output_attention=False,
        # Classification strategy (for classification tasks)
        class_strategy="projection",
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len)

        # Store parameters
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.class_strategy = class_strategy
        self.enc_in = enc_in

        # Embedding - inverted embedding
        # Maps [B, L, N] -> [B, N, E]
        self.enc_embedding = DataEmbedding_inverted(
            seq_len,
            d_model,
            embed,
            freq,
            dropout,
        )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Projection to prediction length
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Embedding: B L N -> B N E (each variate becomes a token)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection: B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        flatten_output=False,
    ):
        # Note: x_dec and x_mark_dec are accepted for compatibility
        dec_out, attns = self.forecast(x_enc, x_mark_enc)

        output = dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if flatten_output:
            output = output.reshape(output.shape[0], -1)

        if self.output_attention:
            return output, attns
        else:
            return output
