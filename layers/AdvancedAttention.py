import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils.masking import TriangularCausalMask

# ================================================================
# 1. Flash Attention (Base - Lấy từ ChatGPT vì xử lý mask tốt)
# ================================================================
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, Dv = values.shape
        
        # Logic tối ưu: Dùng FlashAttn nếu không có mask custom phức tạp
        use_sdp = hasattr(F, "scaled_dot_product_attention") and (attn_mask is None)
        
        if use_sdp:
            q = queries.permute(0, 2, 1, 3) # [B, H, L, E]
            k = keys.permute(0, 2, 1, 3)
            v = values.permute(0, 2, 1, 3)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.mask_flag
            )
            out = out.transpose(1, 2).contiguous().view(B, L, -1)
            return out, None # FlashAttn tiết kiệm bộ nhớ nên không trả map

        # Fallback về Attention thường nếu cần mask phức tạp
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous().view(B, L, -1), A if self.output_attention else None

# ================================================================
# 2. DeepSeek MLA (Multi-Head Latent Attention) - SOTA 2025
# ================================================================
class DS_MultiHeadLatentAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, n_heads=8, mixture_factor=4):
        super().__init__()
        # Interface chuẩn TSLib yêu cầu các tham số init này, dù ta có thể không dùng hết
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads
        self.output_attention = output_attention
        
        # Compression logic
        self.kv_latent_dim = d_model // mixture_factor
        self.query_projection = nn.Linear(d_model, d_model)
        
        # Nén KV
        self.kv_down_proj = nn.Linear(d_model, self.kv_latent_dim)
        self.key_up_proj = nn.Linear(self.kv_latent_dim, d_model)
        self.value_up_proj = nn.Linear(self.kv_latent_dim, d_model)
        
        self.out_projection = nn.Linear(d_model, d_model)
        self.scale = self.d_keys ** -0.5
        self.mask_flag = mask_flag

    def forward(self, queries, keys, values, attn_mask=None, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        
        # 1. Query bình thường
        Q = self.query_projection(queries).view(B, L, self.n_heads, -1)
        
        # 2. DeepSeek Logic: Nén Key/Value
        # Input keys/values [B, S, H, E] -> gộp lại [B, S, D_model]
        # (Giả định keys/values input là state trước khi chia head)
        # Trong TSLib input đã chia head [B, L, H, E], ta reshape lại để nén
        kv_input = keys.reshape(B, S, -1) 
        
        latent = self.kv_down_proj(kv_input) # Nén
        
        K = self.key_up_proj(latent).view(B, S, self.n_heads, -1) # Bung ra
        V = self.value_up_proj(latent).view(B, S, self.n_heads, -1)
        
        # 3. Tính toán
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.scale
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            
        A = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhls,bshd->blhd", A, V).reshape(B, L, -1)
        
        return self.out_projection(out), A if self.output_attention else None

# ================================================================
# 3. Mamba-2 (SSM) Wrapper - Thay thế Attention bằng SSM
# ================================================================
class MambaAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512):
        super().__init__()
        self.output_attention = output_attention
        try:
            from mamba_ssm import Mamba
            # Mamba chỉ cần d_model, các tham số khác nó tự lo
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.valid = True
        except ImportError:
            print("Warning: 'mamba_ssm' not installed. Fallback to Identity.")
            self.mamba = nn.Identity()
            self.valid = False

    def forward(self, queries, keys, values, attn_mask=None, **kwargs):
        # Mamba là mô hình sequence-to-sequence, nó xử lý luồng input (queries)
        # Nó tự quản lý state (keys/values implicit)
        if self.valid:
            # Input của Mamba cần [B, L, D] (đã merge heads)
            B, L, H, E = queries.shape
            x = queries.reshape(B, L, -1) 
            out = self.mamba(x)
            # Reshape lại để khớp với output expected của TSLib [B, L, H, E] 
            # (Lưu ý: Mamba output [B, L, D], ta giả lập lại head)
            out = out.view(B, L, H, E) 
        else:
            out = queries
            
        return out, None # Mamba không có Attention Map