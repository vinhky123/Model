# Model Size Benchmark - Weather Dataset (PowerShell)
# Train models and measure checkpoint file size

$DATASET = "Weather"
$ROOT_PATH = "./dataset/Weather/"
$DATA_PATH = "weather.csv"
$SEQ_LEN = 96
$PRED_LEN = 96
$EPOCHS = 10

# Output CSV file
$OUTPUT_FILE = "model_size_benchmark.csv"

# Create/Clear output file
"model,checkpoint_size_mb,params_count" | Out-File -FilePath $OUTPUT_FILE -Encoding utf8

# List of models to benchmark
$MODELS = @("TimeStar", "TimeXer", "iTransformer", "PatchTST", "Crossformer", "TimesNet", "Nonstationary_Transformer", "LSTM", "Autoformer", "DLinear", "GRU")

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Model Size Benchmark on Weather" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

foreach ($MODEL in $MODELS) {
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host "[$MODEL] Training..." -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    
    # Train model
    python run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --root_path $ROOT_PATH `
      --data_path $DATA_PATH `
      --model_id weather_${SEQ_LEN}_${PRED_LEN} `
      --model $MODEL `
      --data custom `
      --features M `
      --seq_len $SEQ_LEN `
      --label_len 48 `
      --pred_len $PRED_LEN `
      --e_layers 2 `
      --d_layers 1 `
      --factor 3 `
      --enc_in 21 `
      --dec_in 21 `
      --c_out 21 `
      --des 'ModelSize' `
      --d_model 512 `
      --d_ff 512 `
      --batch_size 32 `
      --learning_rate 0.001 `
      --train_epochs $EPOCHS `
      --itr 1
    
    # Find checkpoint file
    $CHECKPOINT_DIR = ".\checkpoints\long_term_forecast_weather_${SEQ_LEN}_${PRED_LEN}_${MODEL}_custom_ftM_sl${SEQ_LEN}_ll48_pl${PRED_LEN}_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_ModelSize_0"
    $CHECKPOINT_FILE = "$CHECKPOINT_DIR\checkpoint.pth"
    
    if (Test-Path $CHECKPOINT_FILE) {
        # Get file size in MB
        $FILE_SIZE = [math]::Round((Get-Item $CHECKPOINT_FILE).Length / 1MB, 2)
        
        # Count parameters using Python
        $PARAM_COUNT = python -c @"
import torch
checkpoint = torch.load('$CHECKPOINT_FILE', map_location='cpu')
total_params = sum(p.numel() for p in checkpoint.values() if hasattr(p, 'numel'))
print(f'{total_params/1e6:.2f}')
"@ 2>$null
        
        if (-not $PARAM_COUNT) { $PARAM_COUNT = "N/A" }
        
        Write-Host ""
        Write-Host "✅ [$MODEL] Checkpoint size: $FILE_SIZE MB" -ForegroundColor Green
        Write-Host "   Parameters: ${PARAM_COUNT}M" -ForegroundColor Cyan
        Write-Host ""
        
        # Save to CSV
        "$MODEL,$FILE_SIZE,$PARAM_COUNT" | Out-File -FilePath $OUTPUT_FILE -Append -Encoding utf8
        
        # Clean up checkpoint directory
        Write-Host "🗑️  Cleaning up checkpoint directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $CHECKPOINT_DIR -ErrorAction SilentlyContinue
    }
    else {
        Write-Host "⚠️  [$MODEL] Checkpoint not found!" -ForegroundColor Red
        "$MODEL,N/A,N/A" | Out-File -FilePath $OUTPUT_FILE -Append -Encoding utf8
    }
    
    # Clear GPU cache
    Write-Host "🧹 Clearing GPU cache..." -ForegroundColor Yellow
    python -c @"
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('   GPU cache cleared')
"@ 2>$null
    
    Write-Host ""
    Start-Sleep -Seconds 2
}

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "✅ All models benchmarked!" -ForegroundColor Green
Write-Host "📊 Results saved to: $OUTPUT_FILE" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""

# Display results
if (Test-Path $OUTPUT_FILE) {
    Write-Host "📈 Summary:" -ForegroundColor Cyan
    Get-Content $OUTPUT_FILE | ForEach-Object { Write-Host $_ }
}

