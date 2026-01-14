# 🎨 Simple Demo - Time Series Forecasting

Script Python đơn giản để chạy inference và visualize kết quả.

## ✨ Features

- ✅ Load model từ checkpoint
- ✅ Chạy inference trên 1 sample bất kỳ
- ✅ Plot input (96 timesteps) + prediction (96 timesteps) + ground truth (96 timesteps)
- ✅ Tính metrics (MAE, MSE, RMSE)
- ✅ Save figure ra file PNG
- ✅ Đơn giản, không cần web server

## 🚀 Cách chạy

### Cơ bản (sample 0, channel 0):
```bash
python simple_demo.py
```

### Chọn sample cụ thể:
```bash
python simple_demo.py --sample_idx 10
```

### Chọn sample và channel:
```bash
python simple_demo.py --sample_idx 5 --channel 2
```

### Chọn model và dataset:
```bash
python simple_demo.py --model TimeXer --data ETTh1 --sample_idx 0
```

## 📊 Arguments

```
--model          Model name (default: TimeStar)
--data           Dataset name (default: ETTm2)
                 Options: ETTh1, ETTh2, ETTm1, ETTm2, weather
--sample_idx     Sample index (default: 0)
                 Range: 0 to N-1 (N = number of test samples)
--channel        Channel to visualize (default: 0)
                 Range: 0 to C-1 (C = number of channels)
--find_best      Find top-k samples with lowest MSE (flag)
--top_k          Number of best samples to show (default: 10)
--visualize_best Visualize all top-k best samples (flag)
```

## 📁 Checkpoint Location

Script tự động tìm checkpoint ở:
```
./checkpoints/long_term_forecast_{model_id}_{model}_{data}_*/checkpoint.pth
```

Ví dụ:
```
./checkpoints/long_term_forecast_ETTm2_96_96_TimeStar_ETTm2_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth
```

## 📈 Output

### Terminal Output:
```
============================================================
🎨 Time Series Forecasting Demo
============================================================

============================================================
Loading Model: TimeStar (Dataset: ETTm2)
============================================================
✅ Found checkpoint: ./checkpoints/...checkpoint.pth
📦 Loading weights from checkpoint...
✅ Model loaded successfully!

📊 Loading test data...
✅ Loaded 2881 test samples

🚀 Running inference on sample 0...
✅ Inference completed!

📈 Metrics (averaged across all channels):
   MAE:  0.1234
   MSE:  0.0567
   RMSE: 0.2381

📊 Plotting results for channel 0...
💾 Saved plot to: demo_output_sample0_channel0.png

============================================================
✅ Demo completed successfully!
============================================================
```

### Plot Output:
- **File**: `demo_output_sample{idx}_channel{ch}.png`
- **Blue line**: Input sequence (96 timesteps)
- **Green line**: Model prediction (96 timesteps)
- **Red dashed**: Ground truth (96 timesteps)
- **Vertical line**: Prediction start point
- **Title**: Includes metrics (MAE, MSE, RMSE)

## 🔍 Find Best Samples

### Find top 10 samples with lowest MSE:
```bash
python simple_demo.py --find_best
```

Output:
```
============================================================
🔍 Finding Top 10 Best Samples (Lowest MSE)
============================================================
Evaluating 2881 samples...
   Processed 100/2881 samples...
   Processed 200/2881 samples...
   ...

============================================================
🏆 Top 10 Samples with Lowest MSE
============================================================
Rank   Sample   MSE          MAE          RMSE        
------------------------------------------------------------
1      1523     0.042318     0.156234     0.205712
2      0847     0.045123     0.162341     0.212453
3      2134     0.048567     0.168923     0.220379
4      0234     0.051234     0.175612     0.226341
5      1876     0.053421     0.179234     0.231234
6      0456     0.055678     0.183456     0.235912
7      1298     0.057234     0.187234     0.239234
8      2456     0.059123     0.191234     0.243123
9      0789     0.061234     0.195234     0.247456
10     1567     0.063456     0.199456     0.251923

📊 Statistics:
   Best MSE:    0.042318 (Sample 1523)
   Worst MSE:   2.345678 (Sample 789)
   Average MSE: 0.234567
   Median MSE:  0.198765
```

### Find top 20 best samples:
```bash
python simple_demo.py --find_best --top_k 20
```

### Find and visualize top 5 best samples:
```bash
python simple_demo.py --find_best --top_k 5 --visualize_best
```
→ Will show charts for all 5 best samples

### Find best samples for specific model/dataset:
```bash
python simple_demo.py --model TimeXer --data ETTh1 --find_best
```

## 🎯 Examples

### Example 1: Default run
```bash
python simple_demo.py
```
→ TimeStar on ETTm2, sample 0, channel 0

### Example 2: Compare different samples
```bash
python simple_demo.py --sample_idx 0
python simple_demo.py --sample_idx 100
python simple_demo.py --sample_idx 500
```

### Example 3: Check different channels
```bash
python simple_demo.py --channel 0
python simple_demo.py --channel 1
python simple_demo.py --channel 2
```

### Example 4: Different models
```bash
python simple_demo.py --model TimeStar --data ETTm2
python simple_demo.py --model TimeXer --data ETTm2
python simple_demo.py --model iTransformer --data ETTm2
```

### Example 5: Different datasets
```bash
python simple_demo.py --model TimeStar --data ETTh1
python simple_demo.py --model TimeStar --data ETTm1
python simple_demo.py --model TimeStar --data weather
```

## 🐛 Troubleshooting

### Error: "No checkpoint found"
```bash
# Check available checkpoints
ls checkpoints/

# Make sure you trained the model first
python run.py --task_name long_term_forecast --model TimeStar ...
```

### Error: "Sample index out of range"
```bash
# Check how many test samples available
python simple_demo.py --sample_idx 0

# Output will show: "Loaded X test samples"
# Valid range: 0 to X-1
```

### Error: "Channel out of range"
```bash
# ETTm2/ETTh1/ETTh2/ETTm1 have 7 channels: 0-6
# Weather has 21 channels: 0-20
```

### Error: "No module named matplotlib"
```bash
pip install matplotlib
```

## 💡 Use Cases

### 1. Quick model verification
```bash
python simple_demo.py --model TimeStar --sample_idx 0
```
→ Check if model inference works

### 2. Qualitative analysis
```bash
for i in 0 10 50 100 500; do
    python simple_demo.py --sample_idx $i
done
```
→ Visualize multiple samples

### 3. Channel comparison
```bash
for ch in 0 1 2 3 4 5 6; do
    python simple_demo.py --channel $ch
done
```
→ Compare all channels

### 4. Model comparison
```bash
python simple_demo.py --model TimeStar --sample_idx 10
python simple_demo.py --model TimeXer --sample_idx 10
python simple_demo.py --model PatchTST --sample_idx 10
```
→ Compare different models on same sample

## 🔍 Code Structure

```python
1. SimpleArgs          # Configuration class
2. find_checkpoint()   # Find checkpoint directory
3. load_model()        # Load model from checkpoint
4. load_data()         # Load test dataset
5. run_inference()     # Run inference on 1 sample
6. plot_results()      # Visualize and save plot
7. main()              # Main entry point
```

## 📝 Dependencies

```bash
pip install torch numpy matplotlib pandas
```

## 🎨 Customization

### Change sequence/prediction length:
Edit `SimpleArgs` class:
```python
self.seq_len = 96   # Input length
self.pred_len = 96  # Prediction length
```

### Change plot style:
Edit `plot_results()` function:
```python
plt.style.use('seaborn')  # Use seaborn style
plt.figure(figsize=(20, 8))  # Larger figure
```

### Add more metrics:
Edit `run_inference()` function:
```python
mape = np.mean(np.abs((prediction - ground_truth) / ground_truth)) * 100
print(f"   MAPE: {mape:.2f}%")
```

---

**Enjoy the simple demo! 🎉**

