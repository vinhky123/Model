#!/bin/bash

# CPU Inference Benchmark for Kaggle
# Simplified version for Kaggle environment

MODEL="TimeStar"
SEQ_LEN=96
PRED_LEN=96
EPOCHS=10

echo "=================================="
echo "CPU Inference Benchmark on Kaggle"
echo "Model: $MODEL"
echo "=================================="

# ETTh1
echo -e "\n[1/5] Running ETTh1..."
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model $MODEL \
  --data ETTh1 \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
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
  --train_epochs $EPOCHS \
  --cpu_inference_benchmark \
  --itr 1

# ETTh2
echo -e "\n[2/5] Running ETTh2..."
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${SEQ_LEN}_${PRED_LEN} \
  --model $MODEL \
  --data ETTh2 \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
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
  --train_epochs $EPOCHS \
  --cpu_inference_benchmark \
  --itr 1

# ETTm1
echo -e "\n[3/5] Running ETTm1..."
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${SEQ_LEN}_${PRED_LEN} \
  --model $MODEL \
  --data ETTm1 \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
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
  --train_epochs $EPOCHS \
  --cpu_inference_benchmark \
  --itr 1

# ETTm2
echo -e "\n[4/5] Running ETTm2..."
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${SEQ_LEN}_${PRED_LEN} \
  --model $MODEL \
  --data ETTm2 \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
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
  --train_epochs $EPOCHS \
  --cpu_inference_benchmark \
  --itr 1

# Weather
echo -e "\n[5/5] Running Weather..."
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Weather/ \
  --data_path weather.csv \
  --model_id weather_${SEQ_LEN}_${PRED_LEN} \
  --model $MODEL \
  --data custom \
  --features M \
  --seq_len $SEQ_LEN \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs $EPOCHS \
  --cpu_inference_benchmark \
  --itr 1

echo -e "\n=================================="
echo "✅ All benchmarks completed!"
echo "📊 Results: cpu_inference_benchmark.csv"
echo "=================================="

# Display results
if [ -f "cpu_inference_benchmark.csv" ]; then
    echo -e "\n📈 Summary:"
    cat cpu_inference_benchmark.csv
fi

