# 🐍 TimeStarMamba: TimeStar with Mamba Architecture

## 📖 Overview

**TimeStarMamba** replaces the O(L²) FlashAttention in TimeStar with O(L) Mamba selective state space model, providing:
- ⚡ **Faster training** - Linear complexity vs quadratic
- 💾 **Lower memory** - Efficient for long sequences
- 🎯 **Selective modeling** - Learns to focus on important information

## 🏗️ Architecture

### Original TimeStar:
```
Input → Embedding → [Self-Attention + STAR Cross-Attention] → Output
                     ↑ O(L²) complexity
```

### TimeStarMamba:
```
Input → Embedding → [Mamba + STAR Cross-Attention] → Output
                     ↑ O(L) complexity
```

**Key Changes:**
- ✅ **MambaLayer** replaces FlashAttention (self-attention)
- ✅ **STAR_patch** kept unchanged (cross-attention with exogenous variables)
- ✅ Same embedding and forecasting head

## 🔧 Numerical Stability Improvements

### Issue:
Mamba's selective scan can produce NaN/Inf values due to:
- Exponential operations in state space discretization
- Gradient accumulation in sequential scan
- Extreme parameter values (A, B, C, Δ)

### Solutions Applied:

#### 1. **MambaLayer Stability** (Line 55-77)
```python
class MambaLayer(nn.Module):
    def __init__(self, configs):
        super(MambaLayer, self).__init__()
        self.mamba = MambaBlock(configs, d_inner, dt_rank)
        self.norm = nn.LayerNorm(configs.d_model)  # ← Stabilize output
        
    def forward(self, q, k, v, attn_mask=None, tau=None, delta=None):
        output = self.mamba(q)
        output = self.norm(output)  # ← Normalize
        output = torch.clamp(output, min=-10, max=10)  # ← Clamp extremes
        return output, None
```

#### 2. **STAR_patch Stability** (Line 19-52)
```python
def forward(self, input, ex_input, *args, **kwargs):
    # ... FFN ...
    combined_mean = self.gen2(combined_mean)
    combined_mean = torch.clamp(combined_mean, min=-10, max=10)  # ← Prevent overflow
    
    if self.training:
        ratio = F.softmax(combined_mean, dim=1)
        
        # Check for NaN/Inf
        if torch.isnan(ratio).any() or torch.isinf(ratio).any():
            print("Warning: NaN/Inf detected, using uniform distribution")
            ratio = torch.ones_like(ratio) / ratio.shape[1]
        
        # Ensure positive and normalized
        ratio = torch.clamp(ratio, min=1e-8)
        ratio = ratio / ratio.sum(dim=1, keepdim=True)
        
        indices = torch.multinomial(ratio, 1)  # Now safe!
    # ...
```

## 🚀 Usage

### Basic Training:
```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id TimeStarMamba_test \
  --model TimeStarMamba \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 128 \
  --d_core 64 \
  --expand 2 \
  --d_conv 4 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --train_epochs 10
```

### Training Speed Benchmark:
```bash
bash scripts/benchmark_timestar_vs_mamba.sh
```

This will compare:
- TimeStar (FlashAttention) O(L²)
- TimeStarMamba (Mamba) O(L)

Results saved to `result_summary.csv`

## 📊 Expected Benefits

| Metric | TimeStar | TimeStarMamba | Improvement |
|--------|----------|---------------|-------------|
| **Training Speed** | Baseline | **1.5-2.5x faster** | ⚡ |
| **Memory Usage** | O(L²) | **O(L)** | 💾 |
| **Long Sequences** | Slow | **Fast** | 🚀 |
| **Accuracy** | ✓ | **~Similar** | 🎯 |

*Actual speedup depends on sequence length - longer sequences = bigger advantage*

## ⚙️ Hyperparameters

### Recommended Settings:

**For stability:**
- `--learning_rate 0.001` (lower than TimeStar's 0.01)
- `--expand 2` (Mamba expansion factor)
- `--d_conv 4` (Conv1d kernel size)
- `--d_model 128` (start small, increase if stable)

**For long sequences:**
- `--seq_len 192` or `--seq_len 336`
- `--batch_size 4` (adjust based on memory)

**Advanced:**
- `--gradient_clip 1.0` (add if instability persists)
- `--warmup_epochs 1` (gradual learning rate warmup)

## 🐛 Troubleshooting

### If you still see NaN/Inf errors:

1. **Lower learning rate:**
```bash
--learning_rate 0.0001
```

2. **Reduce model size:**
```bash
--d_model 64 --expand 1
```

3. **Enable CUDA error details:**
```bash
export CUDA_LAUNCH_BLOCKING=1
python run.py ...
```

4. **Check for NaN in training:**
The model will print warnings if NaN/Inf detected in STAR_patch.

5. **Disable stochastic pooling (temporary):**
Set `model.eval()` during training (not recommended for final results).

## 📈 Performance Tips

### For maximum speed:
- Use longer sequences (seq_len >= 192) to leverage O(L) advantage
- Use GPU with Tensor Cores (A100, H100)
- Install optimized `mamba-ssm` package if available

### For maximum accuracy:
- Start with TimeStar hyperparameters
- Lower learning rate by 10x
- Add gradient clipping
- Use longer warmup period

## 🔍 Key Files

- `models/TimeStarMamba.py` - Main model implementation
- `models/MambaSimple.py` - Mamba block implementation
- `scripts/long_term_forecast/ETT_script/TimeStarMamba.sh` - Training scripts
- `scripts/benchmark_timestar_vs_mamba.sh` - Benchmark script

## 📚 References

1. **Mamba Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
2. **TimeStar Paper:** Check original TimeStar documentation
3. **Implementation:** Based on [mamba-minimal](https://github.com/johnma2006/mamba-minimal/)

## 🎓 Understanding Mamba

See detailed explanation of Mamba's mechanism in the previous response or read:
- Selective State Space Models
- Input-dependent parameters (Δ, B, C)
- Sequential scan with O(L) complexity
- Hardware-aware design

## ✅ Validation Checklist

Before running experiments:
- [ ] Installed all dependencies (`torch`, `einops`)
- [ ] MambaSimple.py is in models directory
- [ ] TimeStarMamba registered in exp_basic.py
- [ ] Dataset paths are correct
- [ ] GPU memory sufficient for batch size

## 🤝 Contributing

If you find issues or improvements:
1. Test with small model first (`d_model=64`)
2. Check for numerical stability
3. Compare with TimeStar baseline
4. Document any hyperparameter changes

---

**Happy Experimenting! 🚀**


