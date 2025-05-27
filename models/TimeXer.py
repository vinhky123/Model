import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
from einops import rearrange
import math


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout, J=1):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.J = J  # Số lượng token Jumbo cho mỗi biến

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Khởi tạo glb_token với J token thay vì 1 token
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, J, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        # Repeat glb_token cho batch size, giờ là [bs, n_vars, J, d_model]
        glb = self.glb_token.repeat(x.shape[0], 1, 1, 1)

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # Ghép x với glb, giờ dim=2 sẽ là patch_num + J
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        J=1,
        jumbo_mlp_ratio=4.0,
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.J = J  # Số lượng token Jumbo
        self.jumbo_dim = J * d_model  # Kích thước ghép của Jumbo token

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Jumbo MLP cho global token
        self.jumbo_mlp = nn.Sequential(
            nn.Linear(self.jumbo_dim, int(self.jumbo_dim * jumbo_mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.jumbo_dim * jumbo_mlp_ratio), self.jumbo_dim),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B = cross.shape[0]  # Batch size gốc
        n_vars = (
            x.shape[0] // B
        )  # Số biến, vì x là [bs * n_vars, patch_num + J, d_model]

        # Self-attention trên tất cả token (patch + J global token)
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        # Cross-attention cho J global token
        x_glb_ori = x[:, -self.J :, :]  # [bs * n_vars, J, d_model]
        x_glb_attn = self.dropout(
            self.cross_attention(
                x_glb_ori, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # FFN riêng cho patch token
        x_patches = x[:, : -self.J, :]  # [bs * n_vars, patch_num, d_model]
        y_patches = self.dropout(
            self.activation(self.conv1(x_patches.transpose(-1, 1)))
        )
        y_patches = self.dropout(self.conv2(y_patches).transpose(-1, 1))

        # FFN riêng cho Jumbo global token
        x_glb_flat = rearrange(x_glb, "b l d -> b (l d)")  # [bs * n_vars, J * d_model]
        y_glb_flat = self.jumbo_mlp(x_glb_flat)  # [bs * n_vars, J * d_model]
        y_glb = rearrange(
            y_glb_flat, "b (l d) -> b l d", l=self.J
        )  # [bs * n_vars, J, d_model]

        # Ghép lại patch token và global token
        y = torch.cat(
            [y_patches, y_glb], dim=1
        )  # [bs * n_vars, patch_num + J, d_model]
        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == "MS" else configs.enc_in
        # Thêm tham số cho Jumbo token
        self.J = getattr(configs, "J", 1)  # Mặc định J=1 nếu không có trong configs
        self.jumbo_mlp_ratio = getattr(configs, "jumbo_mlp_ratio", 4.0)  # Mặc định 4.0

        # Embedding với J
        self.en_embedding = EnEmbedding(
            self.n_vars, configs.d_model, self.patch_len, configs.dropout, self.J
        )
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder với EncoderLayer đã thêm J và jumbo_mlp_ratio
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    J=self.J,
                    jumbo_mlp_ratio=self.jumbo_mlp_ratio,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Cập nhật head_nf vì giờ có patch_num + J thay vì patch_num + 1
        self.head_nf = configs.d_model * (self.patch_num + self.J)
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            if self.features == "M":
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len :, :]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len :, :]
        else:
            return None
