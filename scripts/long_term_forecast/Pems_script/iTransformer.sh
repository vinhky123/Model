export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/PEMS_BAY \
  --data_path pems_bay.csv \
  --model_id pems_bay_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 325 \
  --dec_in 325 \
  --c_out 325 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/PEMS_BAY \
  --data_path pems_bay.csv \
  --model_id traffic_96_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 325 \
  --dec_in 325 \
  --c_out 325 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/PEMS_BAY \
  --data_path pems_bay.csv \
  --model_id traffic_96_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 36 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 325 \
  --dec_in 325 \
  --c_out 325 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/PEMS_BAY \
  --data_path pems_bay.csv \
  --model_id traffic_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 325 \
  --dec_in 325 \
  --c_out 325 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1