model_name=LSTNet

# Weather 96
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

# Weather 192
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

# Weather 336
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1

# Weather 720
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
