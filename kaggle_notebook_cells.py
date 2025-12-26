"""
Kaggle Notebook Cells - Copy và paste từng cell vào Kaggle

Cell này để chạy trong Kaggle Notebook (không phải terminal)
"""

# ============================================================================
# CELL 1: Setup và Check Environment
# ============================================================================
import os
import torch

print("=" * 80)
print("🔧 Environment Setup")
print("=" * 80)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  No GPU available")

# Check directory
print(f"\n📁 Current Directory: {os.getcwd()}")
print(f"📂 Files:")
for item in os.listdir(".")[:10]:
    print(f"   - {item}")

# Check dataset
if os.path.exists("./dataset/ETT-small/"):
    print("\n✅ ETT dataset found")
else:
    print("\n⚠️  ETT dataset not found - check path")

if os.path.exists("./dataset/Weather/"):
    print("✅ Weather dataset found")
else:
    print("⚠️  Weather dataset not found - check path")

print("=" * 80)


# ============================================================================
# CELL 2: Run Single Benchmark (ETTh1) - Test trước
# ============================================================================
!python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimeStar \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --cpu_inference_benchmark \
  --itr 1


# ============================================================================
# CELL 3: Check Results
# ============================================================================
import pandas as pd

if os.path.exists("cpu_inference_benchmark.csv"):
    df = pd.read_csv("cpu_inference_benchmark.csv")
    print("\n" + "=" * 80)
    print("📊 CPU Inference Benchmark Results")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\n" + "=" * 80)
    
    # Summary statistics
    if len(df) > 0:
        print(f"\n📈 Summary:")
        print(f"   Average Latency: {df['latency_per_sample_ms'].mean():.2f} ms/sample")
        print(f"   Average Throughput: {df['throughput_samples_per_sec'].mean():.2f} samples/sec")
        print(f"   Fastest Dataset: {df.loc[df['latency_per_sample_ms'].idxmin(), 'dataset']}")
        print(f"   Slowest Dataset: {df.loc[df['latency_per_sample_ms'].idxmax(), 'dataset']}")
else:
    print("⚠️  No results file found yet")


# ============================================================================
# CELL 4: Run All Benchmarks (nếu test ở Cell 2 OK)
# ============================================================================
!chmod +x kaggle_cpu_benchmark.sh
!./kaggle_cpu_benchmark.sh


# ============================================================================
# CELL 5: Visualize Results (Optional)
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt

if os.path.exists("cpu_inference_benchmark.csv"):
    df = pd.read_csv("cpu_inference_benchmark.csv")
    
    # Plot 1: Latency comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(df['dataset'], df['latency_per_sample_ms'], color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('Latency (ms/sample)')
    plt.title('CPU Inference Latency per Sample')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Throughput comparison
    plt.subplot(1, 2, 2)
    plt.bar(df['dataset'], df['throughput_samples_per_sec'], color='lightcoral')
    plt.xlabel('Dataset')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('CPU Inference Throughput')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cpu_benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Plot saved as 'cpu_benchmark_results.png'")
else:
    print("⚠️  No results to visualize yet")


# ============================================================================
# CELL 6: Compare with GPU Inference (Optional)
# ============================================================================
"""
Nếu muốn so sánh GPU vs CPU inference speed, uncomment code dưới đây:
"""

# import torch
# import time
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# 
# # Load args (cần modify theo config của mày)
# class Args:
#     task_name = 'long_term_forecast'
#     model = 'TimeStar'
#     data = 'ETTh1'
#     # ... thêm các args khác
# 
# args = Args()
# exp = Exp_Long_Term_Forecast(args)
# 
# # GPU inference benchmark
# test_data, test_loader = exp._get_data(flag='test')
# gpu_results = exp.benchmark_inference(test_loader, n_warmup=10, n_test=100)
# 
# # CPU inference benchmark
# cpu_results = exp.benchmark_cpu_inference(test_loader)
# 
# print(f"GPU Latency: {gpu_results['latency']:.2f} ms/sample")
# print(f"CPU Latency: {cpu_results['latency_per_sample_ms']:.2f} ms/sample")
# print(f"Speedup: {cpu_results['latency_per_sample_ms'] / gpu_results['latency']:.2f}x faster on GPU")


# ============================================================================
# CELL 7: Export Results
# ============================================================================
"""
Download file kết quả về máy local
"""

from IPython.display import FileLink

if os.path.exists("cpu_inference_benchmark.csv"):
    print("📥 Click link below to download results:")
    display(FileLink("cpu_inference_benchmark.csv"))
else:
    print("⚠️  No results file to download")


# ============================================================================
# NOTES:
# ============================================================================
"""
🎯 Workflow:
1. Run Cell 1 để check environment
2. Run Cell 2 để test 1 dataset trước (ETTh1)
3. Run Cell 3 để check kết quả
4. Nếu OK, run Cell 4 để chạy tất cả datasets
5. Run Cell 5 để visualize kết quả
6. Run Cell 7 để download file CSV về

⏱️ Estimated Time:
- Cell 2 (single dataset): ~5-10 minutes
- Cell 4 (all datasets): ~30-50 minutes

💾 Disk Space:
- ~400MB per dataset checkpoint
- ~2GB total for all datasets

🔥 Tips:
- Giảm train_epochs từ 10 → 5 nếu muốn chạy nhanh hơn
- Giảm batch_size nếu bị OOM
- Comment out datasets không cần trong Cell 4
"""

