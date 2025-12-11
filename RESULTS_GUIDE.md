# 📊 Hướng Dẫn Tổng Hợp Kết Quả

## 🎯 Tự Động Lưu Kết Quả

Khi chạy experiments, kết quả sẽ **tự động** được lưu vào:

### 1. **`result_summary.csv`** - File tổng hợp
```csv
model,dataset,seq_len,pred_len,mae,mse,rmse,mape,mspe
LSTM,ETTh1,96,96,0.3450,0.2850,0.5339,0.4521,0.1234
LSTM,ETTh1,96,192,0.3821,0.3156,0.5618,0.4892,0.1456
...
```

### 2. **`result_long_term_forecast.txt`** - File text chi tiết
```
long_term_forecast_ETTh1_96_96_LSTM_...
mse:0.285, mae:0.345, dtw:Not calculated
```

---

## 📈 Phân Tích Kết Quả

### 1️⃣ Xem Tất Cả Kết Quả
```bash
python analyze_results.py --format table
```

### 2️⃣ Xem Pivot Table (Model vs Dataset)
```bash
python analyze_results.py --format pivot
```

### 3️⃣ Tìm Best Model Cho Mỗi Dataset
```bash
python analyze_results.py --format best
```

### 4️⃣ So Sánh Theo Prediction Length
```bash
python analyze_results.py --format pred_len
```

### 5️⃣ So Sánh Toàn Diện Các Models
```bash
python analyze_results.py --compare --metric mse
```

### 6️⃣ Export LaTeX Table
```bash
python analyze_results.py --latex
```

---

## 📊 Visualize Kết Quả

### 1️⃣ Plot Tổng Quan
```bash
python plot_results.py --metric mse --mode comparison
```
**Tạo 4 plots:**
- Bar chart: Hiệu suất trung bình của từng model
- Heatmap: Model vs Dataset
- Line plot: Hiệu suất theo prediction length
- Box plot: Phân phối của từng model

### 2️⃣ Plot Chi Tiết Từng Dataset
```bash
python plot_results.py --metric mse --mode dataset
```

### 3️⃣ Thay Đổi Metric
```bash
python plot_results.py --metric mae --mode comparison
python plot_results.py --metric rmse --mode comparison
```

---

## 📋 Ví Dụ Workflow

### Bước 1: Chạy Experiments
```bash
bash scripts/long_term_forecast/ETT_script/LSTM.sh
```

### Bước 2: Xem Kết Quả Nhanh
```bash
python analyze_results.py --format best
```

**Output:**
```
🏆 Best Model for Each Dataset (by MSE):
dataset  model  seq_len  pred_len    mse     mae
ETTh1    LSTM       96        96  0.2850  0.3450
ETTh2    LSTM       96        96  0.3120  0.3820
...
```

### Bước 3: So Sánh Chi Tiết
```bash
python analyze_results.py --compare --metric mse
```

**Output:**
```
📊 Overall Performance:
         mean    std    min    max  count
LSTM   0.3245  0.052  0.285  0.412     16
...

🏆 Best Overall Model: LSTM (mse=0.3245)
```

### Bước 4: Tạo Visualization
```bash
python plot_results.py --metric mse --mode comparison
```
→ Tạo file `results_comparison_mse.png`

---

## 📁 Cấu Trúc Files

```
Model/
├── result_summary.csv          # ⭐ File CSV tổng hợp (main)
├── result_long_term_forecast.txt  # File text chi tiết
├── results/                    # Folder chứa predictions
│   └── long_term_forecast_*/
│       ├── metrics.npy
│       ├── pred.npy
│       └── true.npy
├── analyze_results.py          # Script phân tích
├── plot_results.py             # Script visualize
└── RESULTS_GUIDE.md            # File này
```

---

## 🔧 Tính Năng Nâng Cao

### 1. Lọc Kết Quả Theo Điều Kiện
```python
import pandas as pd
df = pd.read_csv('result_summary.csv')

# Chỉ xem LSTM trên ETTh1
lstm_etth1 = df[(df['model'] == 'LSTM') & (df['dataset'] == 'ETTh1')]
print(lstm_etth1)

# Chỉ xem pred_len = 96
short_pred = df[df['pred_len'] == 96]
print(short_pred.groupby('model')['mse'].mean())
```

### 2. Custom Analysis
```python
import pandas as pd

df = pd.read_csv('result_summary.csv')

# Tìm best config cho từng model
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    best_row = model_df.loc[model_df['mse'].idxmin()]
    print(f"{model}: MSE={best_row['mse']:.4f} on {best_row['dataset']} pred_len={best_row['pred_len']}")
```

### 3. Export Excel
```python
import pandas as pd

df = pd.read_csv('result_summary.csv')
df.to_excel('results.xlsx', index=False)
print("✅ Exported to Excel!")
```

---

## 💡 Tips

1. **Sau mỗi lần chạy experiments**, file CSV tự động update
2. **Xóa CSV để reset**: `rm result_summary.csv`
3. **Backup kết quả**: `cp result_summary.csv results_backup_$(date +%Y%m%d).csv`
4. **Compare 2 runs**: Backup trước khi chạy experiment mới

---

## 📊 Metrics Giải Thích

| Metric | Ý nghĩa | Càng thấp càng tốt |
|--------|---------|-------------------|
| **MAE** | Mean Absolute Error | ✅ |
| **MSE** | Mean Squared Error | ✅ |
| **RMSE** | Root Mean Squared Error | ✅ |
| **MAPE** | Mean Absolute Percentage Error | ✅ |
| **MSPE** | Mean Squared Percentage Error | ✅ |

---

## ✅ Quick Commands

```bash
# Xem kết quả nhanh
cat result_summary.csv

# Count số experiments
wc -l result_summary.csv

# Xem best MSE
python analyze_results.py --format best

# Plot tất cả
python plot_results.py

# So sánh models
python analyze_results.py --compare
```

