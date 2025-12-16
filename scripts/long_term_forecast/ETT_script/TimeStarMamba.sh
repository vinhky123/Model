#!/bin/bash

# TimeStarMamba: TimeStar with Mamba replacing self-attention
# O(L) complexity instead of O(L²)

export CUDA_VISIBLE_DEVICES=0

model_name=TimeStarMamba

# ETTh1 - Prediction length 96
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
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
  --d_model 128 \
  --d_ff 128 \
  --d_core 64 \
  --expand 2 \
  --d_conv 4 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1

# ETTh1 - Prediction length 192
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_core 64 \
  --expand 2 \
  --d_conv 4 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1

# ETTh1 - Prediction length 336
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_core 64 \
  --expand 2 \
  --d_conv 4 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1

# ETTh1 - Prediction length 720
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_core 64 \
  --expand 2 \
  --d_conv 4 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1

