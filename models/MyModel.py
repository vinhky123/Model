import torch
import torch.nn as nn
import torch.nn.functional as F

# --- from iTransformer ---
from layers.Transformer_EncDec import Encoder as InvertedEncoder
from layers.Transformer_EncDec import EncoderLayer as InvertedEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

# --- from PatchTST ---
from layers.Transformer_EncDec import Encoder as PatchEncoder
from layers.Transformer_EncDec import EncoderLayer as PatchEncoderLayer
from layers.Embed import PatchEmbedding

# FlattenHead, Transpose, etc. can be reused


class Transpose(nn.Module):
    """Use this helper if you want the same transpose trick as in PatchTST."""

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        n_vars: number of variables (channels)
        nf:     total input features for the final linear (e.g. d_model * num_patches)
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [B x n_vars x d_model x patch_num]
        x = self.flatten(x)  # -> [B x n_vars x (d_model * patch_num)]
        x = self.linear(x)  # -> [B x n_vars x target_window]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Example "hybrid" model that:
      1) Applies iTransformer’s inverted embedding + inverted transformer encoder.
      2) Feeds that output into PatchTST’s patch embedding + patch encoder + FlattenHead.
      3) De-normalizes.

    You can adapt the forward(...) logic for the different tasks (forecasting, imputation, etc.).
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ----------------------
        # 1) iTransformer Part
        # ----------------------
        self.inverted_embedding = DataEmbedding_inverted(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
        )

        # iTransformer encoder
        self.inverted_encoder = InvertedEncoder(
            [
                InvertedEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)  # number of encoder layers
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # ----------------------
        # 2) PatchTST Part
        # ----------------------
        padding = stride
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=configs.dropout,
        )
        self.patch_encoder = PatchEncoder(
            [
                PatchEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(
                    configs.e_layers
                )  # You could make them different if you want
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
            ),
        )

        # 3) The head is the same FlattenHead from PatchTST
        #   (but we pick final output window based on the task).
        #   For forecasting tasks: output is [B, pred_len, D]
        #   => FlattenHead produces [B, nvars, pred_len], then we do permute(0,2,1).
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)

        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.head = FlattenHead(
                n_vars=configs.enc_in,
                nf=self.head_nf,
                target_window=configs.pred_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name in ["imputation", "anomaly_detection"]:
            self.head = FlattenHead(
                n_vars=configs.enc_in,
                nf=self.head_nf,
                target_window=configs.seq_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == "classification":
            # Example: classification with patch flatten
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class
            )

    def forward_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Example forward pass for forecasting tasks
        (long_term_forecast, short_term_forecast).
        """
        # ------------------------------------------------
        # (A) iTransformer style normalization
        # ------------------------------------------------
        means = x_enc.mean(dim=1, keepdim=True)  # [B, 1, D]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev  # [B, L, D]

        # ------------------------------------------------
        # (B) Inverted Embedding + Inverted Transformer
        # ------------------------------------------------
        # 1) Embedding
        enc_inverted = self.inverted_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        # 2) iTransformer Encoder
        enc_out, _ = self.inverted_encoder(enc_inverted)  # [B, L, d_model]

        # ------------------------------------------------
        # (C) PatchTST
        # ------------------------------------------------
        # 1) PatchEmbedding wants [B, D, L], so permute
        enc_out = enc_out.permute(0, 2, 1)  # [B, d_model, L]

        # 2) Do patch embedding -> [B*nvars, patch_num, d_model]
        #    (In this code, "n_vars" is effectively 1 if you treat d_model as “channels”.
        #     If your data had real "n_vars" separate from d_model, adapt accordingly.)
        patch_x, n_vars = self.patch_embedding(
            enc_out
        )  # shape ~ [B*n_vars, patch_num, d_model]

        # 3) Patch Encoder
        patch_out, _ = self.patch_encoder(
            patch_x
        )  # same shape [B*n_vars, patch_num, d_model]

        # 4) Reshape back to [B, n_vars, patch_num, d_model]
        patch_out = patch_out.reshape(
            -1, n_vars, patch_out.shape[-2], patch_out.shape[-1]
        )
        # -> [B, n_vars, patch_num, d_model]

        # 5) Re-permute to [B, n_vars, d_model, patch_num]
        patch_out = patch_out.permute(0, 1, 3, 2)

        # 6) Head => [B, n_vars, pred_len]
        dec_out = self.head(patch_out)  # shape: [B, n_vars, pred_len]

        # 7) Permute => [B, pred_len, n_vars]
        dec_out = dec_out.permute(0, 2, 1)

        # ------------------------------------------------
        # (D) De-normalize
        # ------------------------------------------------
        # dec_out is [B, pred_len, D]
        # stdev, means shape: [B, 1, D], so broadcast along length dimension
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        return dec_out  # [B, pred_len, D]

    def forward_imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Example for an imputation task.
        You can replicate a similar logic:
          1) do iTransformer-based normalization w.r.t. masked points
          2) pass through iTransformer
          3) pass through PatchTST
          4) revert normalization
        """
        # (A) iTransformer style masked normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)  # [B, D]
        means = means.unsqueeze(1)  # [B, 1, D]
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0.0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )  # [B, D]
        stdev = stdev.unsqueeze(1)  # [B, 1, D]
        x_enc /= stdev

        # (B) iTransformer
        enc_inverted = self.inverted_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.inverted_encoder(enc_inverted)

        # (C) PatchTST
        enc_out = enc_out.permute(0, 2, 1)
        patch_x, n_vars = self.patch_embedding(enc_out)
        patch_out, _ = self.patch_encoder(patch_x)
        patch_out = patch_out.reshape(
            -1, n_vars, patch_out.shape[-2], patch_out.shape[-1]
        )
        patch_out = patch_out.permute(0, 1, 3, 2)
        dec_out = self.head(patch_out)  # [B, n_vars, seq_len]
        dec_out = dec_out.permute(0, 2, 1)  # => [B, seq_len, n_vars]

        # (D) De-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return dec_out

    def forward_anomaly(self, x_enc):
        """
        Similar approach for anomaly_detection.
        """
        # (A) iTransformer style normal
        means = x_enc.mean(dim=1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # (B) iTransformer
        emb = self.inverted_embedding(x_enc, None)
        enc_out, _ = self.inverted_encoder(emb)

        # (C) PatchTST
        enc_out = enc_out.permute(0, 2, 1)
        patch_x, n_vars = self.patch_embedding(enc_out)
        patch_out, _ = self.patch_encoder(patch_x)
        patch_out = patch_out.reshape(
            -1, n_vars, patch_out.shape[-2], patch_out.shape[-1]
        )
        patch_out = patch_out.permute(0, 1, 3, 2)
        dec_out = self.head(patch_out)  # e.g. [B, n_vars, seq_len]
        dec_out = dec_out.permute(0, 2, 1)  # => [B, seq_len, n_vars]

        # (D) Denormalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return dec_out

    def forward_classification(self, x_enc, x_mark_enc):
        """
        Example classification.
        Instead of FlattenHead we have a linear projection at the end.
        """
        # (A) iTransformer style normalization
        means = x_enc.mean(dim=1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # (B) iTransformer
        emb = self.inverted_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.inverted_encoder(emb)  # [B, L, d_model]

        # (C) PatchTST
        enc_out = enc_out.permute(0, 2, 1)  # [B, d_model, L]
        patch_x, n_vars = self.patch_embedding(enc_out)
        patch_out, _ = self.patch_encoder(patch_x)
        patch_out = patch_out.reshape(
            -1, n_vars, patch_out.shape[-2], patch_out.shape[-1]
        )
        patch_out = patch_out.permute(0, 1, 3, 2)  # [B, n_vars, d_model, patch_num]

        # (D) flatten + final linear for classification
        #     (we assigned self.flatten, self.dropout, self.projection in __init__)
        output = self.flatten(patch_out)  # => [B, n_vars * d_model * patch_num]
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # => [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forward_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # if you want only the last self.pred_len, usually we do
            return dec_out[:, -self.pred_len :, :]  # [B, pred_len, D]

        elif self.task_name == "imputation":
            dec_out = self.forward_imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask
            )
            return dec_out  # [B, L, D]

        elif self.task_name == "anomaly_detection":
            dec_out = self.forward_anomaly(x_enc)
            return dec_out  # [B, L, D]

        elif self.task_name == "classification":
            dec_out = self.forward_classification(x_enc, x_mark_enc)
            return dec_out  # [B, num_class]

        else:
            # default or raise an error
            return None
