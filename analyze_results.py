"""
Analyze and summarize experiment results from result_summary.csv
"""
import pandas as pd
import os
import argparse

def analyze_results(csv_file='result_summary.csv', output_format='table'):
    """
    Analyze experiment results
    
    Args:
        csv_file: path to CSV file with results
        output_format: 'table', 'pivot', or 'best'
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Run some experiments first to generate results!")
        return
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print("=" * 100)
    print(f"📊 Experiment Results Summary ({len(df)} experiments)")
    print("=" * 100)
    
    if output_format == 'table':
        # Show all results
        print("\n📋 All Results:")
        print(df.to_string(index=False))
        
    elif output_format == 'pivot':
        # Pivot table: model x dataset
        print("\n📊 MSE by Model and Dataset (lower is better):")
        pivot_mse = df.pivot_table(values='mse', index='model', columns='dataset', aggfunc='mean')
        print(pivot_mse.to_string())
        
        print("\n📊 MAE by Model and Dataset (lower is better):")
        pivot_mae = df.pivot_table(values='mae', index='model', columns='dataset', aggfunc='mean')
        print(pivot_mae.to_string())
        
    elif output_format == 'best':
        # Find best model for each dataset
        print("\n🏆 Best Model for Each Dataset (by MSE):")
        best_models = df.loc[df.groupby('dataset')['mse'].idxmin()]
        print(best_models[['dataset', 'model', 'seq_len', 'pred_len', 'mse', 'mae']].to_string(index=False))
    
    elif output_format == 'pred_len':
        # Compare by prediction length
        print("\n📊 Results by Prediction Length:")
        for pred_len in sorted(df['pred_len'].unique()):
            print(f"\n--- pred_len = {pred_len} ---")
            subset = df[df['pred_len'] == pred_len]
            print(subset[['model', 'dataset', 'mse', 'mae']].to_string(index=False))
    
    # Statistics
    print("\n" + "=" * 100)
    print("📈 Statistics:")
    print(f"  - Total experiments: {len(df)}")
    print(f"  - Models tested: {df['model'].nunique()} ({', '.join(df['model'].unique())})")
    print(f"  - Datasets: {df['dataset'].nunique()} ({', '.join(df['dataset'].unique())})")
    print(f"  - Prediction lengths: {sorted(df['pred_len'].unique())}")
    print("=" * 100)
    
    return df


def compare_models(csv_file='result_summary.csv', metric='mse'):
    """
    Compare all models across all settings
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    print("=" * 100)
    print(f"🔍 Model Comparison by {metric.upper()} (lower is better)")
    print("=" * 100)
    
    # Average across all settings
    model_avg = df.groupby('model')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
    model_avg = model_avg.sort_values('mean')
    
    print("\n📊 Overall Performance:")
    print(model_avg.to_string())
    
    # Best model
    best_model = model_avg.index[0]
    print(f"\n🏆 Best Overall Model: {best_model} ({metric}={model_avg.loc[best_model, 'mean']:.4f})")
    
    return model_avg


def export_latex_table(csv_file='result_summary.csv', output_file='results_table.tex'):
    """
    Export results as LaTeX table
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Create pivot table
    pivot = df.pivot_table(
        values=['mse', 'mae'], 
        index=['model', 'dataset'], 
        columns='pred_len',
        aggfunc='mean'
    )
    
    # Export to LaTeX
    latex_str = pivot.to_latex(float_format="%.4f")
    
    with open(output_file, 'w') as f:
        f.write(latex_str)
    
    print(f"✅ LaTeX table exported to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--file', type=str, default='result_summary.csv', help='CSV file with results')
    parser.add_argument('--format', type=str, default='table', 
                        choices=['table', 'pivot', 'best', 'pred_len'],
                        help='Output format')
    parser.add_argument('--compare', action='store_true', help='Compare models')
    parser.add_argument('--latex', action='store_true', help='Export LaTeX table')
    parser.add_argument('--metric', type=str, default='mse', 
                        choices=['mse', 'mae', 'rmse', 'mape', 'mspe'],
                        help='Metric for comparison')
    
    args = parser.parse_args()
    
    if args.latex:
        export_latex_table(args.file)
    elif args.compare:
        compare_models(args.file, args.metric)
    else:
        analyze_results(args.file, args.format)

