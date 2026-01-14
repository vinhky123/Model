"""
Demo Web App - Time Series Forecasting
Backend Flask server for model inference
"""

from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
import pandas as pd
import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

app = Flask(__name__)

# Global variables to store model and data
MODEL = None
DATA_LOADER = None
ARGS = None

class InferenceArgs:
    """Arguments for inference"""
    def __init__(self):
        # Basic config
        self.task_name = 'long_term_forecast'
        self.is_training = 0
        self.model_id = 'demo'
        self.model = 'TimeStar'  # Default model
        
        # Data loader
        self.data = 'ETTm2'
        self.root_path = './dataset/ETT-small/'
        self.data_path = 'ETTm2.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 't'
        self.checkpoints = './params/'
        
        # Forecasting task
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = 'Monthly'
        self.inverse = False
        
        # Model define
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 96
        
        # GPU
        self.use_gpu = False  # Use CPU for demo
        self.gpu = 0
        self.gpu_type = 'cuda'
        self.use_multi_gpu = False
        self.devices = '0'
        self.device = torch.device('cpu')
        
        # De-stationary projector params
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2
        
        # Optimization (not used in inference)
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 1
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'demo'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False
        
        # Additional
        self.use_dtw = False
        self.d_core = 64
        self.patch_len = 16
        self.individual = False
        
        # TimeFilter specific
        self.alpha = 0.1
        self.top_p = 0.5
        self.pos = 1
        
        # Augmentation (not used)
        self.augmentation_ratio = 0
        self.seed = 2021
        self.channel_reduction_ratio = 0.0


def load_model(model_name='TimeStar'):
    """Load trained model from checkpoint"""
    global MODEL, ARGS
    
    print(f"Loading model: {model_name}")
    
    # Create args
    args = InferenceArgs()
    args.model = model_name
    
    # Find checkpoint directory
    checkpoint_pattern = f"long_term_forecast_*_{model_name}_*"
    checkpoint_dir = Path(args.checkpoints)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoints}")
    
    # Find matching checkpoint
    checkpoint_dirs = list(checkpoint_dir.glob(checkpoint_pattern))
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint found for model: {model_name}")
    
    checkpoint_path = checkpoint_dirs[0] / 'checkpoint.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Create experiment
    exp = Exp_Long_Term_Forecast(args)
    
    # Load checkpoint
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    exp.model.eval()
    
    MODEL = exp.model
    ARGS = args
    
    print(f"Model {model_name} loaded successfully!")
    return exp


def load_data():
    """Load test data"""
    global DATA_LOADER, ARGS
    
    if ARGS is None:
        raise RuntimeError("Model not loaded yet!")
    
    # Create data loader
    from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
    
    if 'ETTh' in ARGS.data:
        dataset = Dataset_ETT_hour
    elif 'ETTm' in ARGS.data:
        dataset = Dataset_ETT_minute
    else:
        raise ValueError(f"Unsupported dataset: {ARGS.data}")
    
    data_set = dataset(
        root_path=ARGS.root_path,
        data_path=ARGS.data_path,
        flag='test',
        size=[ARGS.seq_len, ARGS.label_len, ARGS.pred_len],
        features=ARGS.features,
        target=ARGS.target,
        timeenc=1 if ARGS.embed == 'timeF' else 0,
        freq=ARGS.freq
    )
    
    DATA_LOADER = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    print(f"Data loaded: {len(data_set)} samples")
    return data_set


def run_inference(sample_idx=0):
    """Run inference on a specific sample"""
    global MODEL, DATA_LOADER, ARGS
    
    if MODEL is None or DATA_LOADER is None:
        raise RuntimeError("Model or data not loaded!")
    
    # Get sample
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(DATA_LOADER):
        if i == sample_idx:
            break
    
    # Prepare input
    batch_x = batch_x.float()
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float()
    batch_y_mark = batch_y_mark.float()
    
    # Decoder input
    dec_inp = torch.zeros_like(batch_y[:, -ARGS.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :ARGS.label_len, :], dec_inp], dim=1).float()
    
    # Inference
    with torch.no_grad():
        outputs = MODEL(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # Extract predictions
    f_dim = -1 if ARGS.features == 'MS' else 0
    outputs = outputs[:, -ARGS.pred_len:, f_dim:]
    batch_y = batch_y[:, -ARGS.pred_len:, f_dim:]
    
    # Convert to numpy
    input_seq = batch_x[0].detach().cpu().numpy()  # [seq_len, channels]
    prediction = outputs[0].detach().cpu().numpy()  # [pred_len, channels]
    ground_truth = batch_y[0].detach().cpu().numpy()  # [pred_len, channels]
    
    return input_seq, prediction, ground_truth


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    checkpoint_dir = Path('./params/')
    if not checkpoint_dir.exists():
        return jsonify({'models': []})
    
    models = set()
    for path in checkpoint_dir.glob('long_term_forecast_*'):
        # Extract model name from directory
        parts = path.name.split('_')
        if len(parts) >= 4:
            model_name = parts[3]
            models.add(model_name)
    
    return jsonify({'models': sorted(list(models))})


@app.route('/api/load_model', methods=['POST'])
def load_model_api():
    """Load model endpoint"""
    try:
        data = request.json
        model_name = data.get('model_name', 'TimeStar')
        
        exp = load_model(model_name)
        load_data()
        
        return jsonify({
            'success': True,
            'message': f'Model {model_name} loaded successfully',
            'num_samples': len(DATA_LOADER.dataset)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference', methods=['POST'])
def inference_api():
    """Run inference endpoint"""
    try:
        data = request.json
        sample_idx = data.get('sample_idx', 0)
        channel_idx = data.get('channel_idx', 0)
        
        # Run inference
        input_seq, prediction, ground_truth = run_inference(sample_idx)
        
        # Extract specific channel
        input_data = input_seq[:, channel_idx].tolist()
        pred_data = prediction[:, channel_idx].tolist()
        gt_data = ground_truth[:, channel_idx].tolist()
        
        # Calculate metrics
        mae = np.mean(np.abs(prediction[:, channel_idx] - ground_truth[:, channel_idx]))
        mse = np.mean((prediction[:, channel_idx] - ground_truth[:, channel_idx]) ** 2)
        rmse = np.sqrt(mse)
        
        return jsonify({
            'success': True,
            'input': input_data,
            'prediction': pred_data,
            'ground_truth': gt_data,
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse)
            },
            'seq_len': ARGS.seq_len,
            'pred_len': ARGS.pred_len,
            'num_channels': input_seq.shape[1]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get current model info"""
    if ARGS is None:
        return jsonify({'loaded': False})
    
    return jsonify({
        'loaded': True,
        'model': ARGS.model,
        'dataset': ARGS.data,
        'seq_len': ARGS.seq_len,
        'pred_len': ARGS.pred_len,
        'num_channels': ARGS.enc_in,
        'num_samples': len(DATA_LOADER.dataset) if DATA_LOADER else 0
    })


if __name__ == '__main__':
    print("="*80)
    print("Time Series Forecasting Demo Server")
    print("="*80)
    print("\n🚀 Starting server at http://localhost:5000")
    print("\n📝 Instructions:")
    print("  1. Open browser at http://localhost:5000")
    print("  2. Select a model from dropdown")
    print("  3. Click 'Load Model'")
    print("  4. Adjust sample index and channel")
    print("  5. Click 'Run Inference' to see predictions")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

