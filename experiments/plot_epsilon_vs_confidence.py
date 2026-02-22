#!/usr/bin/env python3
"""
Plot epsilon vs original confidence chosen by MedSecure agent.
Shows that the agent adaptively selects higher epsilon for harder images.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats


def load_data(results_dir: Path):
    """Load epsilon_per_sample and confidence_per_sample from evaluation results."""
    data = {}
    
    # Try loading from individual medsecure_*.json files
    for dataset in ['pathmnist', 'bloodmnist', 'dermamnist']:
        filepath = results_dir / dataset / f"medsecure_{dataset}.json"
        if not filepath.exists():
            filepath = results_dir / f"medsecure_{dataset}.json"
        
        if filepath.exists():
            with open(filepath) as f:
                result = json.load(f)
                if 'epsilon_per_sample' in result and 'confidence_per_sample' in result:
                    data[dataset] = {
                        'epsilon': result['epsilon_per_sample'],
                        'confidence': result['confidence_per_sample']
                    }
                    print(f"Loaded {len(data[dataset]['epsilon'])} samples from {dataset}")
    
    # Fallback: try evaluation_results.json
    if not data:
        eval_file = results_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                results = json.load(f)
                for key, value in results.items():
                    if 'medsecure' in key.lower():
                        if 'epsilon_per_sample' in value and 'confidence_per_sample' in value:
                            dataset = key.replace('medsecure_', '').replace('MedSecure_', '')
                            data[dataset] = {
                                'epsilon': value['epsilon_per_sample'],
                                'confidence': value['confidence_per_sample']
                            }
                            print(f"Loaded {len(data[dataset]['epsilon'])} samples from {dataset}")
    
    return data


def plot_scatter(data: dict, output_path: Path):
    """Plot epsilon vs confidence scatter with regression."""
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = {'pathmnist': '#2ecc71', 'bloodmnist': '#e74c3c', 'dermamnist': '#3498db'}
    labels = {'pathmnist': 'PathMNIST (trained)', 
              'bloodmnist': 'BloodMNIST (zero-shot)', 
              'dermamnist': 'DermaMNIST (zero-shot)'}
    
    for dataset, values in data.items():
        eps = np.array(values['epsilon'])
        conf = np.array(values['confidence'])
        color = colors.get(dataset, '#333333')
        label = labels.get(dataset, dataset)
        
        # Scatter plot with transparency
        ax.scatter(conf, eps, c=color, alpha=0.4, s=20, label=label, edgecolors='none')
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(conf, eps)
        x_line = np.linspace(conf.min(), conf.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2, linestyle='--', alpha=0.8)
        
        print(f"{dataset}: r={r_value:.3f}, p={p_value:.4f}, slope={slope:.4f}")
    
    ax.set_xlabel('Original model confidence', fontsize=12)
    ax.set_ylabel(r'$\epsilon$ (perturbation magnitude)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_kde_2d(data: dict, output_path: Path):
    """Plot 2D KDE for epsilon vs confidence - single dataset."""
    from scipy.stats import gaussian_kde
    
    # Use PathMNIST as primary
    dataset = 'pathmnist' if 'pathmnist' in data else list(data.keys())[0]
    values = data[dataset]
    
    eps = np.array(values['epsilon'])
    conf = np.array(values['confidence'])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 2D KDE
    xy = np.vstack([conf, eps])
    kde = gaussian_kde(xy, bw_method=0.2)
    
    # Grid for evaluation
    x_grid = np.linspace(conf.min() - 0.05, conf.max() + 0.05, 100)
    y_grid = np.linspace(eps.min() - 0.002, eps.max() + 0.005, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    # Contour plot
    contour = ax.contourf(X, Y, Z, levels=20, cmap='Greens', alpha=0.8)
    ax.scatter(conf, eps, c='#2ecc71', alpha=0.3, s=15, edgecolors='none')
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(conf, eps)
    x_line = np.linspace(conf.min(), conf.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black', linewidth=2, linestyle='--', 
            label=f'Linear fit (r={r_value:.2f})')
    
    ax.set_xlabel('Original model confidence', fontsize=12)
    ax.set_ylabel(r'$\epsilon$ (perturbation magnitude)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Density', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_combined(data: dict, output_path: Path):
    """Plot scatter with marginal KDEs - publication quality."""
    from scipy.stats import gaussian_kde
    
    fig = plt.figure(figsize=(7, 5))
    
    # Main scatter plot
    ax_main = fig.add_axes([0.15, 0.15, 0.6, 0.6])
    
    # Marginal axes
    ax_top = fig.add_axes([0.15, 0.77, 0.6, 0.15], sharex=ax_main)
    ax_right = fig.add_axes([0.77, 0.15, 0.15, 0.6], sharey=ax_main)
    
    colors = {'pathmnist': '#2ecc71', 'bloodmnist': '#e74c3c', 'dermamnist': '#3498db'}
    labels = {'pathmnist': 'PathMNIST (trained)', 
              'bloodmnist': 'BloodMNIST (zero-shot)', 
              'dermamnist': 'DermaMNIST (zero-shot)'}
    
    all_conf = []
    all_eps = []
    
    for dataset, values in data.items():
        eps = np.array(values['epsilon'])
        conf = np.array(values['confidence'])
        color = colors.get(dataset, '#333333')
        label = labels.get(dataset, dataset)
        
        all_conf.extend(conf)
        all_eps.extend(eps)
        
        # Main scatter
        ax_main.scatter(conf, eps, c=color, alpha=0.5, s=25, label=label, edgecolors='none')
        
        # Top marginal KDE (confidence)
        conf_kde = gaussian_kde(conf, bw_method=0.2)
        x_conf = np.linspace(min(all_conf) - 0.05, max(all_conf) + 0.05, 200)
        ax_top.fill_between(x_conf, conf_kde(x_conf), alpha=0.4, color=color)
        ax_top.plot(x_conf, conf_kde(x_conf), color=color, linewidth=1.5)
        
        # Right marginal KDE (epsilon)
        eps_kde = gaussian_kde(eps, bw_method=0.15)
        y_eps = np.linspace(min(all_eps) - 0.002, max(all_eps) + 0.005, 200)
        ax_right.fill_betweenx(y_eps, eps_kde(y_eps), alpha=0.4, color=color)
        ax_right.plot(eps_kde(y_eps), y_eps, color=color, linewidth=1.5)
    
    # Overall regression
    all_conf = np.array(all_conf)
    all_eps = np.array(all_eps)
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_conf, all_eps)
    x_line = np.linspace(all_conf.min(), all_conf.max(), 100)
    y_line = slope * x_line + intercept
    ax_main.plot(x_line, y_line, color='black', linewidth=2, linestyle='--', 
                 label=f'Trend (r={r_value:.2f})')
    
    # Formatting
    ax_main.set_xlabel('Original model confidence', fontsize=12)
    ax_main.set_ylabel(r'$\epsilon$ (perturbation magnitude)', fontsize=12)
    ax_main.legend(fontsize=9, loc='upper left')
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # Hide marginal labels
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    # Print stats
    print(f"\nOverall correlation: r={r_value:.3f}, p={p_value:.4f}")
    for dataset, values in data.items():
        eps = np.array(values['epsilon'])
        conf = np.array(values['confidence'])
        r, p = stats.pearsonr(conf, eps)
        print(f"{dataset}: r={r:.3f}, p={p:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot epsilon vs confidence')
    parser.add_argument('--results-dir', type=str, default='results/paper_evaluation',
                        help='Directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                        help='Output directory for figures')
    parser.add_argument('--plot-type', type=str, default='combined',
                        choices=['scatter', 'kde2d', 'combined'],
                        help='Type of plot to generate')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(results_dir)
    
    if not data:
        print(f"ERROR: No data found in {results_dir}")
        print("Expected files with 'epsilon_per_sample' and 'confidence_per_sample'")
        return
    
    # Generate plots
    if args.plot_type == 'scatter':
        plot_scatter(data, output_dir / 'epsilon_vs_confidence.png')
    elif args.plot_type == 'kde2d':
        plot_kde_2d(data, output_dir / 'epsilon_vs_confidence_kde.png')
    else:  # combined
        plot_combined(data, output_dir / 'epsilon_vs_confidence.png')


if __name__ == '__main__':
    main()