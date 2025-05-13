import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Tái sử dụng các lớp từ code của mày
class Transpose(nn.Module):
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


# Các lớp mới cho Swin Transformer
class PatchEmbed2D(nn.Module):
    """Chia không gian 2D (seq_len x n_vars) thành patch"""

    def __init__(self, seq_len, n_vars, patch_size, d_model, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.patch_size = patch_size  # (height, width)
        self.grid_size = (seq_len // patch_size[0], n_vars // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 1, seq_len, n_vars]
        B, C, H, W = x.shape
        assert (
            H == self.seq_len and W == self.n_vars
        ), f"Input size ({H}x{W}) doesn't match ({self.seq_len}x{self.n_vars})."

        x = self.proj(x)  # [B, d_model, grid_size[0], grid_size[1]]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        x = self.dropout(x)
        return x, self.n_vars


class WindowAttention2D(nn.Module):
    """Window-based Multi-head Self-Attention cho 2D"""

    def __init__(self, dim, window_size, num_heads, shift_size=(0, 0), dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (height, width)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.shift_size = shift_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, f"Number of patches ({N}) doesn't match H*W ({H}*{W})."

        x = x.view(B, H, W, C)
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)
            )

        # Chia thành windows
        x = x.view(
            B,
            H // self.window_size[0],
            self.window_size[0],
            W // self.window_size[1],
            self.window_size[1],
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size[0] * self.window_size[1], C)

        # Tính Q, K, V
        qkv = self.qkv(x).reshape(
            -1,
            self.window_size[0] * self.window_size[1],
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(-1, self.window_size[0] * self.window_size[1], C)
        )
        x = self.proj(x)
        x = self.dropout(x)

        # Dựng lại
        x = x.view(
            B,
            H // self.window_size[0],
            W // self.window_size[1],
            self.window_size[0],
            self.window_size[1],
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(
                x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )

        x = x.view(B, N, C)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=(0, 0), dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(dim, window_size, num_heads, shift_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class SwinEncoder(nn.Module):
    def __init__(self, configs, window_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SwinBlock(
                    dim=configs.d_model,
                    num_heads=configs.n_heads,
                    window_size=window_size,
                    shift_size=(
                        (window_size[0] // 2, window_size[1] // 2)
                        if i % 2 == 1
                        else (0, 0)
                    ),
                    dropout=configs.dropout,
                )
                for i in range(configs.e_layers)
            ]
        )
        self.norm = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
        )

    def forward(self, x, H, W):
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.norm(x)
        return x, None  # Trả None cho attns để khớp với Encoder gốc


class Model(nn.Module):
    """
    PatchTST với Swin Transformer 2D, giữ flow của PatchTST gốc
    Paper link (PatchTST): https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding (theo seq_len)
        stride: int, stride for patch_embedding (theo seq_len)
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model

        # Patch embedding 2D
        patch_size = (patch_len, 1)  # Chỉ cắt theo seq_len, n_vars giữ nguyên
        window_size = (7, self.n_vars)  # Window bao quát hết n_vars
        self.patch_embedding = PatchEmbed2D(
            self.seq_len, self.n_vars, patch_size, configs.d_model, configs.dropout
        )

        # Encoder
        self.encoder = SwinEncoder(configs, window_size=window_size)

        # Prediction Head
        self.grid_size = self.patch_embedding.grid_size
        self.head_nf = configs.d_model * self.grid_size[0] * self.grid_size[1]
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.head = FlattenHead(
                self.n_vars, self.head_nf, self.pred_len, head_dropout=configs.dropout
            )
        elif self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.head = FlattenHead(
                self.n_vars, self.head_nf, self.seq_len, head_dropout=configs.dropout
            )
        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * self.n_vars, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Reshape thành 2D
        x_enc = x_enc.view(-1, 1, self.seq_len, self.n_vars)  # [B, 1, seq_len, n_vars]

        # Patch embedding
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B, num_patches, d_model]

        # Encoder
        H, W = self.grid_size
        enc_out, _ = self.encoder(enc_out, H, W)  # [B, num_patches, d_model]

        # Reshape để khớp với FlattenHead
        enc_out = enc_out.view(-1, n_vars, H, self.d_model).permute(
            0, 1, 3, 2
        )  # [B, n_vars, d_model, patch_num]

        # Head
        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, n_vars]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
