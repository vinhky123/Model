"""
Plot experiment results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison(csv_file='result_summary.csv', metric='mse', save_fig=True):
    """
    Plot comparison of models across datasets and prediction lengths
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Performance Comparison ({metric.upper()})', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar plot - Average performance by model
    ax1 = axes[0, 0]
    model_avg = df.groupby('model')[metric].mean().sort_values()
    model_avg.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title(f'Average {metric.upper()} by Model', fontsize=12, fontweight='bold')
    ax1.set_xlabel(metric.upper())
    ax1.set_ylabel('Model')
    
    # Plot 2: Heatmap - Model vs Dataset
    ax2 = axes[0, 1]
    pivot = df.pivot_table(values=metric, index='model', columns='dataset', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': metric.upper()})
    ax2.set_title(f'{metric.upper()}: Model vs Dataset', fontsize=12, fontweight='bold')
    
    # Plot 3: Line plot - Performance by prediction length
    ax3 = axes[1, 0]
    for model in df['model'].unique():
        model_data = df[df['model'] == model].groupby('pred_len')[metric].mean()
        ax3.plot(model_data.index, model_data.values, marker='o', label=model, linewidth=2)
    ax3.set_title(f'{metric.upper()} by Prediction Length', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Prediction Length')
    ax3.set_ylabel(metric.upper())
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot - Distribution by model
    ax4 = axes[1, 1]
    df_plot = df[['model', metric]].copy()
    df_plot.boxplot(by='model', ax=ax4, patch_artist=True)
    ax4.set_title(f'{metric.upper()} Distribution by Model', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Model')
    ax4.set_ylabel(metric.upper())
    plt.sca(ax4)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_fig:
        output_file = f'results_comparison_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {output_file}")
    
    plt.show()


def plot_dataset_comparison(csv_file='result_summary.csv', metric='mse'):
    """
    Plot detailed comparison for each dataset
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    datasets = df['dataset'].unique()
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]
        
        # Group by model and pred_len
        pivot = dataset_df.pivot_table(values=metric, index='pred_len', columns='model', aggfunc='mean')
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Prediction Length')
        ax.set_ylabel(metric.upper())
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = f'results_by_dataset_{metric}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {output_file}")
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('--file', type=str, default='result_summary.csv', help='CSV file')
    parser.add_argument('--metric', type=str, default='mse', 
                        choices=['mse', 'mae', 'rmse', 'mape', 'mspe'],
                        help='Metric to plot')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['comparison', 'dataset'],
                        help='Plot mode')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison':
        plot_comparison(args.file, args.metric)
    elif args.mode == 'dataset':
        plot_dataset_comparison(args.file, args.metric)

