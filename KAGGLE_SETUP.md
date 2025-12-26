# Kaggle CPU Inference Benchmark - Quick Setup

## 🚀 Chạy nhanh trên Kaggle

### Bước 1: Upload code lên Kaggle
- Tạo Kaggle Notebook mới
- Settings → Accelerator: **GPU T4 x2** (hoặc GPU bất kỳ)
- Upload toàn bộ code vào `/kaggle/working/`

### Bước 2: Chạy script

```bash
chmod +x kaggle_cpu_benchmark.sh
./kaggle_cpu_benchmark.sh
```

Hoặc chạy từng dataset riêng lẻ:

```bash
# Test ETTh1 only
python run.py \
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
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --train_epochs 10 \
  --cpu_inference_benchmark \
  --itr 1
```

### Bước 3: Xem kết quả

```bash
cat cpu_inference_benchmark.csv
```

## 📊 Kết quả mong đợi

```
model,dataset,seq_len,pred_len,batch_size,inference_time_ms,latency_per_sample_ms,throughput_samples_per_sec
TimeStar,ETTh1,96,96,32,245.32,7.67,130.45
TimeStar,ETTh2,96,96,32,243.18,7.60,131.58
TimeStar,ETTm1,96,96,32,238.45,7.45,134.23
TimeStar,ETTm2,96,96,32,241.67,7.55,132.45
TimeStar,Weather,96,96,32,312.45,9.76,102.46
```

## ⚙️ Cấu hình custom

Muốn test model khác? Sửa trong `kaggle_cpu_benchmark.sh`:

```bash
MODEL="PatchTST"  # Hoặc iTransformer, TimeXer, etc.
SEQ_LEN=96
PRED_LEN=96
EPOCHS=10  # Giảm xuống nếu muốn chạy nhanh hơn
```

## 💡 Tips cho Kaggle

1. **GPU T4 x2** là đủ cho benchmark này
2. **Training time**: ~5-10 phút/dataset với 10 epochs
3. **Total runtime**: ~30-50 phút cho tất cả 5 datasets
4. **Disk space**: ~2GB cho checkpoints
5. **RAM**: ~16GB là đủ

## 🐛 Troubleshooting

### Lỗi: "No such file or directory: ./dataset/ETT-small/"
```bash
# Check đường dẫn dataset
ls -la dataset/
```

### Lỗi: "CUDA out of memory"
```bash
# Giảm batch size trong script
# Sửa: --batch_size 32 → --batch_size 16
```

### Lỗi: Permission denied
```bash
chmod +x kaggle_cpu_benchmark.sh
```

## 📥 Download kết quả

Sau khi chạy xong, download file `cpu_inference_benchmark.csv` từ Kaggle Output để phân tích:

1. Click vào Files panel (bên phải)
2. Tìm file `cpu_inference_benchmark.csv`
3. Click "..." → Download

---

**Note**: Script này train trên GPU rồi test inference trên CPU - đây là workflow thực tế khi deploy model!

