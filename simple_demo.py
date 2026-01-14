"""
Simple Demo - Time Series Forecasting
Load model, run inference on 1 sample, and visualize

Usage:
    python simple_demo.py --sample_idx 0
    python simple_demo.py --sample_idx 10 --channel 0
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from data_provider.data_factory import data_provider


class SimpleArgs:
    """Simple arguments for inference"""
    def __init__(self, model_name='TimeStar', data_name='ETTm2'):
        # Basic config
        self.task_name = 'long_term_forecast'
        self.is_training = 0
        self.model_id = f'{data_name}_96_96'
        self.model = model_name
        
        # Data loader
        self.data = data_name
        if 'ETT' in data_name:
            self.root_path = './dataset/ETT-small/'
            self.data_path = f'{data_name}.csv'
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
        elif data_name.lower() == 'weather':
            self.root_path = './dataset/Weather/'
            self.data_path = 'weather.csv'
            self.enc_in = 21
            self.dec_in = 21
            self.c_out = 21
        else:
            raise ValueError(f"Unsupported dataset: {data_name}")
        
        self.features = 'M'
        self.target = 'OT'
        self.freq = 't' if 'ETTm' in data_name else 'h'
        self.checkpoints = './checkpoints/'
        
        # Forecasting task
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = 'Monthly'
        self.inverse = False
        
        # Model define (default values, will be overridden by checkpoint)
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
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
        self.use_gpu = False  # Use CPU for simplicity
        self.gpu = 0
        self.gpu_type = 'cuda'
        self.use_multi_gpu = False
        self.devices = '0'
        self.device = torch.device('cpu')
        
        # Additional
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2
        self.use_dtw = False
        self.d_core = 64
        self.patch_len = 16
        self.individual = False
        self.alpha = 0.1
        self.top_p = 0.5
        self.pos = 1
        self.augmentation_ratio = 0
        self.seed = 2021
        self.channel_reduction_ratio = 0.0
        self.num_workers = 0
        self.batch_size = 1


def parse_config_from_checkpoint_name(checkpoint_dir_name):
    """Parse model config from checkpoint directory name"""
    import re
    
    config = {}
    
    # Extract d_model (dm256 → 256)
    dm_match = re.search(r'_dm(\d+)_', checkpoint_dir_name)
    if dm_match:
        config['d_model'] = int(dm_match.group(1))
    
    # Extract n_heads (nh8 → 8)
    nh_match = re.search(r'_nh(\d+)_', checkpoint_dir_name)
    if nh_match:
        config['n_heads'] = int(nh_match.group(1))
    
    # Extract e_layers (el2 → 2)
    el_match = re.search(r'_el(\d+)_', checkpoint_dir_name)
    if el_match:
        config['e_layers'] = int(el_match.group(1))
    
    # Extract d_layers (dl1 → 1)
    dl_match = re.search(r'_dl(\d+)_', checkpoint_dir_name)
    if dl_match:
        config['d_layers'] = int(dl_match.group(1))
    
    # Extract d_ff (df2048 → 2048)
    df_match = re.search(r'_df(\d+)_', checkpoint_dir_name)
    if df_match:
        config['d_ff'] = int(df_match.group(1))
    
    # Extract expand (expand2 → 2)
    expand_match = re.search(r'_expand(\d+)_', checkpoint_dir_name)
    if expand_match:
        config['expand'] = int(expand_match.group(1))
    
    # Extract d_conv (dc4 → 4)
    dc_match = re.search(r'_dc(\d+)_', checkpoint_dir_name)
    if dc_match:
        config['d_conv'] = int(dc_match.group(1))
    
    # Extract factor (fc3 → 3)
    fc_match = re.search(r'_fc(\d+)_', checkpoint_dir_name)
    if fc_match:
        config['factor'] = int(fc_match.group(1))
    
    return config


def find_checkpoint(model_name, data_name):
    """Find checkpoint directory for given model and data"""
    checkpoint_dir = Path('./checkpoints/')
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Search pattern
    pattern = f"long_term_forecast_*_{model_name}_{data_name}_*"
    
    matching_dirs = list(checkpoint_dir.glob(pattern))
    
    if not matching_dirs:
        # Try with 'custom' for weather
        pattern = f"long_term_forecast_*_{model_name}_custom_*"
        matching_dirs = list(checkpoint_dir.glob(pattern))
    
    if not matching_dirs:
        print(f"\n❌ No checkpoint found for: {model_name} on {data_name}")
        print(f"   Pattern: {pattern}")
        print(f"\n📂 Available checkpoints:")
        for d in checkpoint_dir.glob("long_term_forecast_*"):
            print(f"   - {d.name}")
        raise FileNotFoundError(f"Checkpoint not found")
    
    checkpoint_dir_path = matching_dirs[0]
    checkpoint_path = checkpoint_dir_path / 'checkpoint.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint.pth not found in {checkpoint_dir_path}")
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Parse config from directory name
    config = parse_config_from_checkpoint_name(checkpoint_dir_path.name)
    
    if config:
        print(f"📝 Parsed config from checkpoint:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    return checkpoint_path, config


def load_model(model_name, data_name):
    """Load model from checkpoint"""
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    
    print(f"\n{'='*60}")
    print(f"Loading Model: {model_name} (Dataset: {data_name})")
    print(f"{'='*60}")
    
    # Create args with default config
    args = SimpleArgs(model_name, data_name)
    
    # Find checkpoint and parse config
    checkpoint_path, parsed_config = find_checkpoint(model_name, data_name)
    
    # Update args with parsed config from checkpoint
    if parsed_config:
        for key, value in parsed_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"   Updated {key} = {value}")
    
    # Create experiment with updated config
    exp = Exp_Long_Term_Forecast(args)
    
    # Load checkpoint
    print(f"📦 Loading weights from checkpoint...")
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    exp.model.eval()
    
    print(f"✅ Model loaded successfully!")
    return exp, args


def load_data(args):
    """Load test data"""
    print(f"\n📊 Loading test data...")
    
    data_set, data_loader = data_provider(args, 'test')
    
    print(f"✅ Loaded {len(data_set)} test samples")
    return data_set, data_loader


def run_inference(exp, data_loader, sample_idx):
    """Run inference on specific sample"""
    print(f"\n🚀 Running inference on sample {sample_idx}...")
    
    # Get sample
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        if i == sample_idx:
            break
    else:
        raise ValueError(f"Sample {sample_idx} not found! Dataset has {len(data_loader)} samples")
    
    # Prepare input
    batch_x = batch_x.float()
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float()
    batch_y_mark = batch_y_mark.float()
    
    # Decoder input
    dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float()
    
    # Inference
    with torch.no_grad():
        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # Extract predictions
    f_dim = -1 if exp.args.features == 'MS' else 0
    outputs = outputs[:, -exp.args.pred_len:, f_dim:]
    batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]
    
    # Convert to numpy
    input_seq = batch_x[0].detach().cpu().numpy()  # [seq_len, channels]
    prediction = outputs[0].detach().cpu().numpy()  # [pred_len, channels]
    ground_truth = batch_y[0].detach().cpu().numpy()  # [pred_len, channels]
    
    # Calculate metrics
    mae = np.mean(np.abs(prediction - ground_truth))
    mse = np.mean((prediction - ground_truth) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"✅ Inference completed!")
    print(f"\n📈 Metrics (averaged across all channels):")
    print(f"   MAE:  {mae:.4f}")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    return input_seq, prediction, ground_truth


def plot_results(input_seq, prediction, ground_truth, channel_idx=0, sample_idx=0):
    """Plot input, prediction, and ground truth"""
    print(f"\n📊 Plotting results for channel {channel_idx}...")
    
    # Extract specific channel
    input_data = input_seq[:, channel_idx]
    pred_data = prediction[:, channel_idx]
    gt_data = ground_truth[:, channel_idx]
    
    # Calculate metrics for this channel
    mae = np.mean(np.abs(pred_data - gt_data))
    mse = np.mean((pred_data - gt_data) ** 2)
    rmse = np.sqrt(mse)
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    seq_len = len(input_data)
    pred_len = len(pred_data)
    
    # Create x-axis
    x_input = np.arange(seq_len)
    x_pred = np.arange(seq_len, seq_len + pred_len)
    
    # Plot input
    plt.plot(x_input, input_data, 'b-', label='Input', linewidth=2)
    
    # Plot prediction
    plt.plot(x_pred, pred_data, 'g-', label='Prediction', linewidth=2)
    
    # Plot ground truth
    plt.plot(x_pred, gt_data, 'r--', label='Ground Truth', linewidth=2)
    
    # Add vertical line at prediction start
    plt.axvline(x=seq_len, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(seq_len, plt.ylim()[1], ' Prediction Start', 
             verticalalignment='top', fontsize=10, color='gray')
    
    # Labels and title
    plt.xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title(f'Time Series Forecasting - Sample {sample_idx}, Channel {channel_idx}\n'
              f'MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}', 
              fontsize=14, fontweight='bold')
    
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure first
    output_file = f'demo_output_sample{sample_idx}_channel{channel_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved plot to: {output_file}")
    
    # Show plot (blocking - will wait for user to close window)
    print(f"📊 Displaying plot... (close window to continue)")
    plt.show()


def find_best_samples(exp, data_loader, top_k=10):
    """Find samples with lowest MSE"""
    print(f"\n{'='*60}")
    print(f"🔍 Finding Top {top_k} Best Samples (Lowest MSE)")
    print(f"{'='*60}")
    print(f"Evaluating {len(data_loader)} samples...")
    
    results = []
    
    # Evaluate all samples
    for sample_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        # Prepare input
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        
        # Decoder input
        dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float()
        
        # Inference
        with torch.no_grad():
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # Extract predictions
        f_dim = -1 if exp.args.features == 'MS' else 0
        outputs = outputs[:, -exp.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]
        
        # Convert to numpy
        prediction = outputs[0].detach().cpu().numpy()
        ground_truth = batch_y[0].detach().cpu().numpy()
        
        # Calculate metrics
        mae = np.mean(np.abs(prediction - ground_truth))
        mse = np.mean((prediction - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        
        results.append({
            'sample_idx': sample_idx,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        })
        
        # Progress indicator
        if (sample_idx + 1) % 100 == 0:
            print(f"   Processed {sample_idx + 1}/{len(data_loader)} samples...")
    
    # Sort by MSE
    results.sort(key=lambda x: x['mse'])
    
    # Display top k
    print(f"\n{'='*60}")
    print(f"🏆 Top {top_k} Samples with Lowest MSE")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Sample':<8} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 60)
    
    for rank, result in enumerate(results[:top_k], 1):
        print(f"{rank:<6} {result['sample_idx']:<8} "
              f"{result['mse']:<12.6f} {result['mae']:<12.6f} {result['rmse']:<12.6f}")
    
    print(f"\n📊 Statistics:")
    print(f"   Best MSE:    {results[0]['mse']:.6f} (Sample {results[0]['sample_idx']})")
    print(f"   Worst MSE:   {results[-1]['mse']:.6f} (Sample {results[-1]['sample_idx']})")
    print(f"   Average MSE: {np.mean([r['mse'] for r in results]):.6f}")
    print(f"   Median MSE:  {np.median([r['mse'] for r in results]):.6f}")
    
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Simple Time Series Forecasting Demo')
    parser.add_argument('--model', type=str, default='TimeStar', help='Model name')
    parser.add_argument('--data', type=str, default='ETTm2', help='Dataset name')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--channel', type=int, default=0, help='Channel to visualize')
    parser.add_argument('--find_best', action='store_true', help='Find top 10 samples with lowest MSE')
    parser.add_argument('--top_k', type=int, default=10, help='Number of best samples to show (default: 10)')
    parser.add_argument('--visualize_best', action='store_true', help='Visualize all top-k best samples')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎨 Time Series Forecasting Demo")
    print("="*60)
    
    try:
        # 1. Load model
        exp, model_args = load_model(args.model, args.data)
        
        # 2. Load data
        data_set, data_loader = load_data(model_args)
        
        # Mode 1: Find best samples
        if args.find_best:
            best_samples = find_best_samples(exp, data_loader, top_k=args.top_k)
            
            # Optionally visualize all best samples
            if args.visualize_best:
                print(f"\n📊 Visualizing top {args.top_k} best samples...")
                for result in best_samples:
                    sample_idx = result['sample_idx']
                    print(f"\n--- Sample {sample_idx} (MSE: {result['mse']:.6f}) ---")
                    
                    # Run inference
                    input_seq, prediction, ground_truth = run_inference(exp, data_loader, sample_idx)
                    
                    # Plot
                    plot_results(input_seq, prediction, ground_truth, args.channel, sample_idx)
            
            print(f"\n{'='*60}")
            print("✅ Best samples search completed!")
            print(f"{'='*60}\n")
            return 0
        
        # Mode 2: Visualize specific sample
        # Validate sample_idx
        if args.sample_idx >= len(data_set):
            print(f"\n⚠️  Sample index {args.sample_idx} out of range!")
            print(f"   Valid range: 0 - {len(data_set)-1}")
            return
        
        # Validate channel
        if args.channel >= model_args.enc_in:
            print(f"\n⚠️  Channel {args.channel} out of range!")
            print(f"   Valid range: 0 - {model_args.enc_in-1}")
            return
        
        # 3. Run inference
        input_seq, prediction, ground_truth = run_inference(exp, data_loader, args.sample_idx)
        
        # 4. Plot results
        plot_results(input_seq, prediction, ground_truth, args.channel, args.sample_idx)
        
        print(f"\n{'='*60}")
        print("✅ Demo completed successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

