#!/bin/bash

# Tạo thư mục checkpoints nếu chưa tồn tại
mkdir -p ./checkpoints

# Chạy training với các tham số mặc định
python run_btc.py \
    --task_name btc_classification \
    --is_training 1 \
    --model_id btc_timesnet \
    --model TimesNet \
    --data_path ./dataset/btc_data.csv \
    --features M \
    --target Close \
    --freq d \
    --checkpoints ./checkpoints \
    --seq_len 96 \
    --pred_len 5 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 2 \
    --num_class 2 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --moving_avg 25 \
    --factor 1 \
    --distil \
    --dropout 0.1 \
    --embed timeF \
    --activation gelu \
    --channel_independence 1 \
    --num_workers 10 \
    --train_epochs 10 \
    --batch_size 32 \
    --patience 3 \
    --learning_rate 0.0001 \
    --loss MSE \
    --lradj type1 \
    --use_gpu True \
    --gpu 0

# Lưu log vào file
echo "Training completed at $(date)" >> ./checkpoints/training_log.txt 