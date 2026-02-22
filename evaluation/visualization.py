"""
Visualization utilities for adversarial robustness evaluation.

Includes functions for plotting metrics, heatmaps, perturbation visualizations,
and generating comprehensive reports.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Visualization will not work.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


def set_style(style: str = 'seaborn-v0_8-whitegrid'):
    """Set matplotlib style."""
    check_matplotlib()
    try:
        plt.style.use(style)
    except:
        plt.style.use('seaborn-whitegrid' if 'seaborn' in plt.style.available else 'ggplot')


class MetricsPlotter:
    """
    Plot attack metrics and comparisons.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        check_matplotlib()
        self.figsize = figsize
        self.dpi = dpi
        set_style()
    
    def plot_asr_comparison(
        self,
        methods: List[str],
        asr_values: List[float],
        errors: Optional[List[float]] = None,
        title: str = "Attack Success Rate Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Bar plot comparing ASR across methods.
        
        Args:
            methods: Method names
            asr_values: ASR values for each method
            errors: Optional error bars
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        x = np.arange(len(methods))
        bars = ax.bar(x, asr_values, color='steelblue', edgecolor='black', alpha=0.8)
        
        if errors is not None:
            ax.errorbar(x, asr_values, yerr=errors, fmt='none', color='black', capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Attack Success Rate')
        ax.set_ylim(0, 1)
        ax.set_title(title)
        
        # Add value labels
        for bar, val in zip(bars, asr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_metrics_radar(
        self,
        methods: List[str],
        metrics: Dict[str, List[float]],
        title: str = "Multi-Metric Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Radar/spider plot for multi-metric comparison.
        
        Args:
            methods: Method names
            metrics: Dict mapping metric name to list of values per method
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        metric_names = list(metrics.keys())
        n_metrics = len(metric_names)
        n_methods = len(methods)
        
        # Angles for radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=self.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        
        for idx, method in enumerate(methods):
            values = [metrics[m][idx] for m in metric_names]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_title(title, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_convergence(
        self,
        iterations: List[int],
        values_dict: Dict[str, List[float]],
        ylabel: str = "Loss",
        title: str = "Attack Convergence",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Line plot showing convergence over iterations.
        
        Args:
            iterations: Iteration numbers
            values_dict: Dict mapping method/run name to list of values
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for name, values in values_dict.items():
            ax.plot(iterations[:len(values)], values, label=name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_epsilon_sweep(
        self,
        epsilons: List[float],
        asr_values: List[float],
        quality_values: Optional[List[float]] = None,
        title: str = "Epsilon vs Attack Success",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ASR and quality metrics vs epsilon.
        
        Args:
            epsilons: Epsilon values
            asr_values: ASR at each epsilon
            quality_values: Optional quality metric (e.g., SSIM)
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        fig, ax1 = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Epsilon (ε)')
        ax1.set_ylabel('Attack Success Rate', color=color1)
        line1 = ax1.plot(epsilons, asr_values, 'o-', color=color1, linewidth=2, label='ASR')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1)
        
        if quality_values is not None:
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('Image Quality (SSIM)', color=color2)
            line2 = ax2.plot(epsilons, quality_values, 's--', color=color2, linewidth=2, label='SSIM')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(0, 1)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right')
        
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig


class PerturbationVisualizer:
    """
    Visualize adversarial perturbations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 4), dpi: int = 100):
        check_matplotlib()
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_perturbation_comparison(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
        amplification: float = 10.0,
        title: str = "Adversarial Example",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Side-by-side comparison of original, adversarial, and perturbation.
        
        Args:
            original: Original image [C, H, W]
            adversarial: Adversarial image [C, H, W]
            perturbation: Optional perturbation (computed if not provided)
            amplification: Factor to amplify perturbation for visibility
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        if perturbation is None:
            perturbation = adversarial - original
        
        # Convert to numpy and transpose to [H, W, C]
        orig_np = original.cpu().numpy().transpose(1, 2, 0)
        adv_np = adversarial.cpu().numpy().transpose(1, 2, 0)
        pert_np = perturbation.cpu().numpy().transpose(1, 2, 0)
        
        # Handle grayscale
        if orig_np.shape[-1] == 1:
            orig_np = orig_np.squeeze(-1)
            adv_np = adv_np.squeeze(-1)
            pert_np = pert_np.squeeze(-1)
        
        # Clip to valid range
        orig_np = np.clip(orig_np, 0, 1)
        adv_np = np.clip(adv_np, 0, 1)
        
        # Amplify and normalize perturbation for visualization
        pert_vis = pert_np * amplification
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
        
        axes[0].imshow(orig_np, cmap='gray' if orig_np.ndim == 2 else None)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(adv_np, cmap='gray' if adv_np.ndim == 2 else None)
        axes[1].set_title('Adversarial')
        axes[1].axis('off')
        
        axes[2].imshow(pert_vis, cmap='hot' if pert_vis.ndim == 2 else None)
        axes[2].set_title(f'Perturbation (×{amplification})')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_perturbation_heatmap(
        self,
        perturbation: torch.Tensor,
        title: str = "Perturbation Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Heatmap showing perturbation magnitude.
        
        Args:
            perturbation: Perturbation tensor [C, H, W]
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        # Compute magnitude across channels
        magnitude = perturbation.abs().sum(dim=0).cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        im = ax.imshow(magnitude, cmap='hot')
        ax.set_title(title)
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Perturbation Magnitude')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_batch_grid(
        self,
        images: torch.Tensor,
        nrow: int = 8,
        title: str = "Image Grid",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a grid of images.
        
        Args:
            images: Batch of images [N, C, H, W]
            nrow: Number of images per row
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        n_images = images.shape[0]
        ncol = nrow
        nrow = (n_images + ncol - 1) // ncol
        
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.5), dpi=self.dpi)
        axes = np.atleast_2d(axes)
        
        for idx in range(nrow * ncol):
            ax = axes[idx // ncol, idx % ncol]
            if idx < n_images:
                img = images[idx].cpu().numpy().transpose(1, 2, 0)
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                img = np.clip(img, 0, 1)
                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig


class ConfusionMatrixPlotter:
    """
    Plot confusion matrices for adversarial attacks.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        check_matplotlib()
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_transition_matrix(
        self,
        true_labels: np.ndarray,
        adversarial_preds: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Class Transition Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot matrix showing where samples move after attack.
        
        Args:
            true_labels: Ground truth labels
            adversarial_preds: Predictions after attack
            class_names: Optional class names
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        n_classes = max(true_labels.max(), adversarial_preds.max()) + 1
        
        # Build transition matrix
        matrix = np.zeros((n_classes, n_classes))
        for true, pred in zip(true_labels, adversarial_preds):
            matrix[true, pred] += 1
        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_norm = np.divide(matrix, row_sums, where=row_sums > 0)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        im = ax.imshow(matrix_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                val = matrix_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        
        if class_names:
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
        
        ax.set_xlabel('Adversarial Prediction')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Transition Probability')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig


class TrainingVisualizer:
    """
    Visualize RL agent training progress.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        check_matplotlib()
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        window_size: int = 100,
        title: str = "Training Progress",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training curves with smoothing.
        
        Args:
            metrics: Dict mapping metric name to list of values over time
            window_size: Window for moving average smoothing
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        axes = axes.flatten()
        
        for idx, (name, values) in enumerate(metrics.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            values = np.array(values)
            
            # Raw values (light)
            ax.plot(values, alpha=0.3, color='blue')
            
            # Smoothed values
            if len(values) > window_size:
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                ax.plot(range(window_size-1, len(values)), smoothed, color='blue', linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_strategy_distribution(
        self,
        strategy_history: List[int],
        strategy_names: List[str],
        window_size: int = 1000,
        title: str = "Strategy Selection Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot how strategy selection evolves during training.
        
        Args:
            strategy_history: List of strategy indices chosen
            strategy_names: Names of strategies
            window_size: Window for computing distribution
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        n_strategies = len(strategy_names)
        n_windows = len(strategy_history) // window_size
        
        distributions = np.zeros((n_windows, n_strategies))
        
        for i in range(n_windows):
            window = strategy_history[i*window_size:(i+1)*window_size]
            for s in window:
                if s < n_strategies:
                    distributions[i, s] += 1
            distributions[i] /= window_size
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        x = np.arange(n_windows) * window_size
        
        ax.stackplot(x, distributions.T, labels=strategy_names, alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Strategy Proportion')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
        
        return fig


class ReportGenerator:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_plotter = MetricsPlotter()
        self.pert_visualizer = PerturbationVisualizer()
        self.confusion_plotter = ConfusionMatrixPlotter()
        self.training_viz = TrainingVisualizer()
    
    def generate_full_report(
        self,
        results: Dict[str, Any],
        method_name: str = "MedSecure",
        include_plots: bool = True
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Dictionary containing all evaluation results
            method_name: Name of the method being evaluated
            include_plots: Whether to generate plot files
        
        Returns:
            Path to generated report
        """
        report_lines = []
        report_lines.append(f"# {method_name} Evaluation Report")
        report_lines.append(f"\n## Summary\n")
        
        # Extract key metrics
        if 'metrics' in results:
            metrics = results['metrics']
            report_lines.append("### Attack Performance")
            report_lines.append(f"- Attack Success Rate: {metrics.get('ASR', 0):.2%}")
            report_lines.append(f"- SSIM: {metrics.get('SSIM', 0):.4f}")
            report_lines.append(f"- PSNR: {metrics.get('PSNR', 0):.2f} dB")
            report_lines.append(f"- L∞ Norm: {metrics.get('L_inf', 0):.4f}")
            if 'LPIPS' in metrics:
                report_lines.append(f"- LPIPS: {metrics.get('LPIPS', 0):.4f}")
        
        if 'efficiency' in results:
            efficiency = results['efficiency']
            report_lines.append("\n### Efficiency")
            report_lines.append(f"- Mean Queries/Attack: {efficiency.get('mean_queries_per_attack', 0):.1f}")
            report_lines.append(f"- Mean Time/Attack: {efficiency.get('mean_time_per_attack', 0):.3f} s")
            report_lines.append(f"- Efficiency Score: {efficiency.get('efficiency_score', 0):.4f}")
        
        if 'diversity' in results:
            diversity = results['diversity']
            report_lines.append("\n### Diversity")
            report_lines.append(f"- Strategy Coverage: {diversity.get('strategy_coverage', 0):.2%}")
            report_lines.append(f"- Strategy Diversity: {diversity.get('strategy_diversity', 0):.4f}")
            report_lines.append(f"- Vulnerability Coverage: {diversity.get('vulnerability_coverage', 0):.2%}")
        
        # Comparison with baselines
        if 'baseline_comparison' in results:
            report_lines.append("\n## Baseline Comparison\n")
            comparison = results['baseline_comparison']
            
            report_lines.append("| Method | ASR | SSIM | Queries |")
            report_lines.append("|--------|-----|------|---------|")
            
            for method, data in comparison.items():
                asr = data.get('ASR', 0)
                ssim = data.get('SSIM', 0)
                queries = data.get('queries', 0)
                report_lines.append(f"| {method} | {asr:.2%} | {ssim:.4f} | {queries:.0f} |")
        
        # Generate plots
        if include_plots and MATPLOTLIB_AVAILABLE:
            report_lines.append("\n## Visualizations\n")
            
            if 'baseline_comparison' in results:
                methods = list(results['baseline_comparison'].keys())
                asr_values = [results['baseline_comparison'][m].get('ASR', 0) for m in methods]
                
                fig = self.metrics_plotter.plot_asr_comparison(
                    methods, asr_values,
                    save_path=str(self.output_dir / 'asr_comparison.png')
                )
                plt.close(fig)
                report_lines.append("![ASR Comparison](asr_comparison.png)")
        
        # Write report
        report_path = self.output_dir / 'report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also save raw results as JSON
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj
            
            json.dump(convert(results), f, indent=2)
        
        return str(report_path)


def create_summary_figure(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['ASR', 'SSIM', 'Queries'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary figure comparing methods.
    
    Args:
        results: Dict mapping method name to metrics dict
        metrics: Metrics to include
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    check_matplotlib()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5), dpi=100)
    
    if n_metrics == 1:
        axes = [axes]
    
    methods = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[m].get(metric, 0) for m in methods]
        
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        
        # Add value labels
        for bar, val in zip(bars, values):
            fmt = '.2%' if metric == 'ASR' else '.2f' if metric == 'SSIM' else '.0f'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:{fmt}}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Method Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=100)
    
    return fig
