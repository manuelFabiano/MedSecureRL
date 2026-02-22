#!/usr/bin/env python3
"""
Plot epsilon distribution chosen by MedSecure agent.
Shows that the agent adaptively selects different epsilon values for different images.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def load_epsilon_data(results_dir: Path):
    """Load epsilon_per_sample from evaluation results."""
    data = {}
    
    for dataset in ['pathmnist', 'dermamnist', 'bloodmnist']:
        filepath = results_dir / f"medsecure_{dataset}.json"
        
        if filepath.exists():
            with open(filepath) as f:
                result = json.load(f)
                if 'epsilon_per_sample' in result:
                    data[dataset] = result['epsilon_per_sample']
                    print(f"Loaded {len(data[dataset])} samples from {dataset}")
    
    return data


def plot_combined(data: dict, output_path: Path):
    """Plot combined figure with KDE for all datasets."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    colors = {'pathmnist': '#2ecc71', 'bloodmnist': '#e74c3c', 'dermamnist': '#3498db'}
    labels = {'pathmnist': 'PathMNIST (trained)', 
              'bloodmnist': 'BloodMNIST (zero-shot)', 
              'dermamnist': 'DermaMNIST (zero-shot)'}
    
    # Find global range for x-axis (with some padding)
    all_eps = np.concatenate([np.array(data[d]) for d in data])
    x_min = max(0, all_eps.min() - 0.002)
    x_max = min(0.035, all_eps.max() + 0.005)
    x_range = np.linspace(x_min, x_max, 500)
    
    for dataset in ['pathmnist', 'dermamnist', 'bloodmnist']:
        if dataset not in data:
            continue
        epsilons = data[dataset]
        eps = np.array(epsilons)
        color = colors.get(dataset, '#333333')
        label = labels.get(dataset, dataset)
        
        # KDE
        kde = stats.gaussian_kde(eps, bw_method=0.15)
        density = kde(x_range)
        
        ax.plot(x_range, density, color=color, linewidth=2.5, label=label)
        ax.fill_between(x_range, density, alpha=0.3, color=color)
        
        # Mean line
        ax.axvline(eps.mean(), color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Budget line
    if x_max >= 0.03:
        ax.axvline(0.03, color='black', linestyle=':', linewidth=2, label='Budget (0.03)')
    
    ax.set_xlabel(r'$\epsilon$ (perturbation magnitude)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot epsilon distribution')
    parser.add_argument('--results-dir', type=str, default='evaluation',
                        help='Directory containing medsecure_*.json files')
    parser.add_argument('--output', type=str, default='fig_epsilon_distribution.png',
                        help='Output filename')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    # Load data
    data = load_epsilon_data(results_dir)
    
    if not data:
        print(f"ERROR: No epsilon data found in {results_dir}")
        return
    
    # Generate plot
    plot_combined(data, output_path)
    
    # Print summary statistics
    print("\n=== Epsilon Statistics ===")
    for dataset, epsilons in data.items():
        eps = np.array(epsilons)
        print(f"{dataset}: mean={eps.mean():.4f}, std={eps.std():.4f}, "
              f"min={eps.min():.4f}, max={eps.max():.4f}")


if __name__ == '__main__':
    main()