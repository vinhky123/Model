#!/bin/bash

# Benchmark: TimeStar vs TimeStarMamba
# Compare training speed of O(L²) attention vs O(L) Mamba

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Training Speed Benchmark"
echo "TimeStar (FlashAttention) vs TimeStarMamba (Mamba)"
echo "=========================================="
echo ""

# Common parameters
root_path="./dataset/ETT-small/"
data_path="ETTh1.csv"
data="ETTh1"
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_model=128
d_ff=128
d_core=64
expand=2
d_conv=4
batch_size=4

echo "=========================================="
echo "1. TimeStar (Original with FlashAttention)"
echo "=========================================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id TimeStar_benchmark \
  --model TimeStar \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Benchmark' \
  --d_model $d_model \
  --d_ff $d_ff \
  --d_core $d_core \
  --expand $expand \
  --d_conv $d_conv \
  --batch_size $batch_size \
  --train_speed_benchmark

echo ""
echo "=========================================="
echo "2. TimeStarMamba (Mamba replacing Attention)"
echo "=========================================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id TimeStarMamba_benchmark \
  --model TimeStarMamba \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Benchmark' \
  --d_model $d_model \
  --d_ff $d_ff \
  --d_core $d_core \
  --expand $expand \
  --d_conv $d_conv \
  --batch_size $batch_size \
  --train_speed_benchmark

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "Check result_summary.csv for detailed comparison"
echo "=========================================="

