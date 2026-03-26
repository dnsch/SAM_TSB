from matplotlib.pyplot import plot
from torch import nn

from src.utils.samformer_utils.attention import scaled_dot_product_attention

# from src.utils.samformer_utils.revin import RevIN

from src.utils.revin import RevIN

from src.base.model import BaseModel


class SAMFormer(BaseModel):
    def __init__(
        self,
        seq_len=96,
        hid_dim=16,
        pred_len=96,
        input_channels=None,
        attn_dropout=0,
        output_dropout=0,
        plot_attention=False,
        **args,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len, input_channels=input_channels)

        # Network architecture:
        # I think there was a bug in the original implementation,
        # see: https://github.com/romilbert/samformer/issues/20
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, hid_dim)
        self.output_layer = nn.Linear(hid_dim, seq_len)
        self.linear_forecaster = nn.Linear(seq_len, pred_len)

        # Dropout layers
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        # Initialize weights to match SAMFormer init
        self._init_weights()

        self.plot_attention = plot_attention
        self.attention_pattern = None

    # TODO: check if this is doing the same as base model function
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot Uniform (TensorFlow default)"""
        for module in [
            self.compute_keys,
            self.compute_queries,
            self.compute_values,
            self.output_layer,
            self.linear_forecaster,
        ]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, flatten_output=False):
        # Note: If use_revin, then x is already revin normalized
        # Channel-Wise Attention
        queries = self.compute_queries(x)  # (n, D, hid_dim)
        keys = self.compute_keys(x)  # (n, D, hid_dim)
        values = self.compute_values(x)  # (n, D, L)

        # Save attention_patterns
        if self.plot_attention:
            att_score, attention_pattern = scaled_dot_product_attention(
                queries, keys, values, plot_attention=self.plot_attention
            )  # (n, D, L)
            self.attention_pattern = attention_pattern
        else:
            att_score = scaled_dot_product_attention(
                queries, keys, values, plot_attention=self.plot_attention
            )  # (n, D, L)

        # apply att dropout
        att_score = self.attn_dropout(att_score)

        # Output layer
        att_score = self.output_layer(att_score)

        # apply output dropout
        att_score = self.output_dropout(att_score)

        out = x + att_score  # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out)  # (n, D, H)
        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])
        else:
            return out
