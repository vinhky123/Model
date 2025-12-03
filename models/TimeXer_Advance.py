import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# Import các module SOTA mới (Giả sử bạn đã lưu file ở bước trước là layers/SOTA_Attention.py)
from layers.AdvancedAttention import FlashAttention, DS_MultiHeadLatentAttention, MambaAttention
import numpy as np

# ================================================================
# GIỮ NGUYÊN CÁC KHỐI PHỤ TRỢ CỦA BẠN
# ================================================================

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
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
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
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # Self Attention Part
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # Cross Attention Part (Global Token mixing)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

# ================================================================
# BASE CLASS: Chứa logic forecast chung để đỡ lặp code
# ================================================================
class BaseTimeXer(nn.Module):
    def __init__(self, configs):
        super(BaseTimeXer, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
        # Embeddings
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Output Head
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def forecast_logic(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        # Encoder call (self.encoder sẽ được định nghĩa ở lớp con)
        enc_out = self.encoder(en_embed, ex_embed)
        
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
             return self.forecast_logic(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        return None

# ================================================================
# MODEL 1: TimeXer_Flash (Sử dụng FlashAttention-3)
# Ưu điểm: Tốc độ cao, chuẩn xác, tiết kiệm bộ nhớ GPU
# ================================================================
class TimeXer_Flash(BaseTimeXer):
    def __init__(self, configs):
        super(TimeXer_Flash, self).__init__(configs)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self Attention: Dùng FlashAttention bọc trong AttentionLayer
                    AttentionLayer(
                        FlashAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    # Cross Attention: Dùng FlashAttention
                    AttentionLayer(
                        FlashAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

# ================================================================
# MODEL 2: TimeXer_DeepSeek (Sử dụng MLA - Multi-Head Latent)
# Ưu điểm: Nén KV Cache cực tốt, thích hợp mô hình lớn
# ================================================================
class TimeXer_DeepSeek(BaseTimeXer):
    def __init__(self, configs):
        super(TimeXer_DeepSeek, self).__init__(configs)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self Attention: Dùng DeepSeek MLA (Thay thế hoàn toàn AttentionLayer)
                    DS_MultiHeadLatentAttention(
                        mask_flag=False, 
                        attention_dropout=configs.dropout, 
                        output_attention=False,
                        d_model=configs.d_model, 
                        n_heads=configs.n_heads,
                        mixture_factor=4 # Hệ số nén của DeepSeek
                    ),
                    # Cross Attention: Vẫn dùng DeepSeek MLA
                    DS_MultiHeadLatentAttention(
                        mask_flag=False, 
                        attention_dropout=configs.dropout, 
                        output_attention=False,
                        d_model=configs.d_model, 
                        n_heads=configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

# ================================================================
# MODEL 3: TimeXer_Jamba (Hybrid Mamba + FlashAttention)
# Ưu điểm: Tuyến tính hóa O(N) cho self-attention, xử lý chuỗi cực dài
# ================================================================
class TimeXer_Jamba(BaseTimeXer):
    def __init__(self, configs):
        super(TimeXer_Jamba, self).__init__(configs)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # Self Attention: Thay thế bằng MAMBA (SSM)
                    # Mamba không cần AttentionLayer bọc ngoài
                    MambaAttention(
                        d_model=configs.d_model,
                        output_attention=False
                    ),
                    # Cross Attention: 
                    # Mamba không giỏi Cross-Attn (Query != Key), nên ta dùng FlashAttention ở đây để "Neo" lại
                    AttentionLayer(
                        FlashAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

# Class mặc định để TSLib gọi nếu cần
class Model(TimeXer_Flash): # Bạn có thể đổi class cha ở đây để chọn model mặc định
    def __init__(self, configs):
        super(Model, self).__init__(configs)