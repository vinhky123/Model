#!/bin/bash

# Model Size Benchmark - Weather Dataset
# Train models and measure checkpoint file size

DATASET="Weather"
ROOT_PATH="./dataset/Weather/"
DATA_PATH="weather.csv"
SEQ_LEN=96
PRED_LEN=96
EPOCHS=10

# Output CSV file
OUTPUT_FILE="model_size_benchmark.csv"

# Create/Clear output file
echo "model,checkpoint_size_mb,params_count" > $OUTPUT_FILE

# List of models to benchmark
MODELS=("TimeStar" "TimeXer" "iTransformer" "PatchTST" "Crossformer" "TimesNet" "Nonstationary_Transformer" "LSTM" "Autoformer" "DLinear" "GRU")

echo "=================================="
echo "Model Size Benchmark on Weather"
echo "=================================="
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$MODEL] Training..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Train model
    python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
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
      --des 'ModelSize' \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --train_epochs $EPOCHS \
      --itr 1
    
    # Find checkpoint file
    CHECKPOINT_DIR="./checkpoints/long_term_forecast_weather_${SEQ_LEN}_${PRED_LEN}_${MODEL}_custom_ftM_sl${SEQ_LEN}_ll48_pl${PRED_LEN}_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_ModelSize_0"
    CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint.pth"
    
    if [ -f "$CHECKPOINT_FILE" ]; then
        # Get file size in MB
        FILE_SIZE=$(du -m "$CHECKPOINT_FILE" | cut -f1)
        
        # Count parameters using Python
        PARAM_COUNT=$(python -c "
import torch
checkpoint = torch.load('$CHECKPOINT_FILE', map_location='cpu')
total_params = sum(p.numel() for p in checkpoint.values() if hasattr(p, 'numel'))
print(f'{total_params/1e6:.2f}')
" 2>/dev/null || echo "N/A")
        
        echo ""
        echo "✅ [$MODEL] Checkpoint size: ${FILE_SIZE} MB"
        echo "   Parameters: ${PARAM_COUNT}M"
        echo ""
        
        # Save to CSV
        echo "$MODEL,$FILE_SIZE,$PARAM_COUNT" >> $OUTPUT_FILE
        
        # Clean up checkpoint directory
        echo "🗑️  Cleaning up checkpoint directory..."
        rm -rf "$CHECKPOINT_DIR"
    else
        echo "⚠️  [$MODEL] Checkpoint not found!"
        echo "$MODEL,N/A,N/A" >> $OUTPUT_FILE
    fi
    
    # Clear GPU cache
    echo "🧹 Clearing GPU cache..."
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('   GPU cache cleared')
" 2>/dev/null
    
    echo ""
    sleep 2
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All models benchmarked!"
echo "📊 Results saved to: $OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Display results
if [ -f "$OUTPUT_FILE" ]; then
    echo "📈 Summary:"
    column -t -s',' $OUTPUT_FILE
fi

