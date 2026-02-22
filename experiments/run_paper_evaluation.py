#!/usr/bin/env python3
"""
Run complete evaluation for the MedSecure paper.

Uses the same evaluation logic as evaluate.py to ensure consistency.
Evaluates MedSecure and baselines (FGSM, PGD, Freq, AutoAttack) on 
multiple datasets with multiple epsilon values.

Usage:
    # Zero-shot transfer (same model for all datasets)
    python experiments/run_paper_evaluation.py \
        --agent results/trained_agent/final_agent \
        --victim-model checkpoints/victim/resnet18_pathmnist_best.pth \
        --datasets pathmnist dermamnist bloodmnist \
        --output-dir results/evaluation \
        --n-samples 500 \
        --include-autoattack

    # With per-dataset models (if available)
    python experiments/run_paper_evaluation.py \
        --agent results/trained_agent/final_agent \
        --victim-model checkpoints/victim/resnet18_pathmnist_best.pth \
        --victim-models-dir checkpoints/victim \
        --datasets pathmnist dermamnist bloodmnist
"""

import argparse
import json
import numpy as np
import torch
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from file to avoid experiments/__init__.py
import importlib.util
_eval_spec = importlib.util.spec_from_file_location(
    "evaluate", 
    Path(__file__).parent / "evaluate.py"
)
_eval_module = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_module)
Evaluator = _eval_module.Evaluator


def run_full_evaluation(
    evaluator: Evaluator,
    n_samples: int,
    baseline_epsilons: List[float],
    medsecure_epsilon: float,
    include_autoattack: bool = False
) -> Dict[str, Any]:
    """
    Run full evaluation using the Evaluator class.
    
    Args:
        evaluator: Configured Evaluator instance
        n_samples: Number of samples to evaluate
        baseline_epsilons: List of epsilon values for baselines
        medsecure_epsilon: Epsilon budget for MedSecure
        include_autoattack: Whether to include AutoAttack
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    # =================================================================
    # PRE-SELECT CORRECTLY CLASSIFIED SAMPLES
    # This ensures ALL methods are evaluated on the SAME samples
    # =================================================================
    print(f"\n{'='*50}")
    print(f"Pre-selecting {n_samples} correctly classified samples...")
    print(f"{'='*50}")
    
    fixed_samples = []
    with torch.no_grad():
        for images, labels in evaluator.test_loader:
            images = images.to(evaluator.device)
            labels = labels.to(evaluator.device).squeeze()
            
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            outputs = evaluator.victim_model(images)
            preds = outputs.argmax(dim=1)
            
            # Keep only correctly classified
            for i in range(images.size(0)):
                if preds[i].item() == labels[i].item():
                    fixed_samples.append((images[i:i+1].clone(), labels[i:i+1].clone()))
                    if len(fixed_samples) >= n_samples:
                        break
            
            if len(fixed_samples) >= n_samples:
                break
    
    print(f"  Selected {len(fixed_samples)} correctly classified samples")
    
    # Store fixed samples in evaluator for use by all methods
    evaluator.fixed_samples = fixed_samples
    
    # =================================================================
    # BASELINE EVALUATION (multiple epsilon values)
    # =================================================================
    baselines = ['fgsm', 'pgd', 'freq']
    if include_autoattack:
        baselines.append('autoattack')
    
    for attack_name in baselines:
        for epsilon in baseline_epsilons:
            print(f"\n{'='*50}")
            print(f"Evaluating {attack_name.upper()} at ε={epsilon}")
            print(f"{'='*50}")
            
            try:
                result = evaluator.evaluate_baseline(
                    attack_name=attack_name,
                    n_samples=n_samples,
                    epsilon=epsilon
                )
                
                key = f"{attack_name}_eps{epsilon:.3f}"
                results[key] = {
                    'method': attack_name.upper(),
                    'epsilon': epsilon,
                    'epsilon_std': 0.0,  # Fixed epsilon
                    'asr': result['metrics']['ASR'],
                    'ssim': result['metrics']['SSIM'],
                    'psnr': result['metrics']['PSNR'],
                    'lpips': result['metrics'].get('LPIPS', 0),
                    'queries': result['efficiency'].get('mean_queries_per_attack', 0),
                    'n_samples': result['n_samples'],
                }
                
                print(f"  ASR:  {result['metrics']['ASR']*100:.1f}%")
                print(f"  SSIM: {result['metrics']['SSIM']:.4f}")
                print(f"  PSNR: {result['metrics']['PSNR']:.2f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # =================================================================
    # MEDSECURE EVALUATION (adaptive epsilon)
    # =================================================================
    print(f"\n{'='*50}")
    print(f"Evaluating MedSecure (adaptive ε, budget={medsecure_epsilon})")
    print(f"{'='*50}")
    
    try:
        medsecure_result = evaluator.evaluate_agent(
            n_samples=n_samples,
            epsilon=medsecure_epsilon
        )
        
        # Extract per-sample epsilon if available
        epsilon_per_sample = medsecure_result.get('epsilon_per_sample', [])
        if epsilon_per_sample:
            mean_eps = np.mean(epsilon_per_sample)
            std_eps = np.std(epsilon_per_sample)
        else:
            mean_eps = medsecure_epsilon
            std_eps = 0.0
        
        results['medsecure'] = {
            'method': 'MedSecure',
            'epsilon': mean_eps,
            'epsilon_std': std_eps,
            'epsilon_budget': medsecure_epsilon,
            'epsilon_per_sample': epsilon_per_sample,
            'asr': medsecure_result['metrics']['ASR'],
            'ssim': medsecure_result['metrics']['SSIM'],
            'psnr': medsecure_result['metrics']['PSNR'],
            'lpips': medsecure_result['metrics'].get('LPIPS', 0),
            'queries': medsecure_result['efficiency'].get('mean_queries_per_attack', 0),
            'n_samples': medsecure_result['n_samples'],
            'strategy_distribution': medsecure_result.get('strategy_distribution', {}),
            'strategy_asr': medsecure_result.get('strategy_asr', {}),
        }
        
        # Add confidence per sample if available
        if 'confidence_per_sample' in medsecure_result:
            results['medsecure']['confidence_per_sample'] = medsecure_result['confidence_per_sample']
        
        print(f"  ASR:  {medsecure_result['metrics']['ASR']*100:.1f}%")
        print(f"  SSIM: {medsecure_result['metrics']['SSIM']:.4f}")
        print(f"  PSNR: {medsecure_result['metrics']['PSNR']:.2f}")
        print(f"  Mean ε: {mean_eps:.4f} ± {std_eps:.4f}")
        if medsecure_result.get('strategy_distribution'):
            print(f"  Strategy: {medsecure_result['strategy_distribution']}")
            
    except Exception as e:
        print(f"  ERROR evaluating MedSecure: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def _save_results_incrementally(all_results: Dict[str, Any], output_dir: Path, timestamp: str):
    """Save results incrementally to avoid data loss on errors."""
    for key, result in all_results.items():
        result_json = {}
        for k, v in result.items():
            if isinstance(v, (np.ndarray, list)):
                if isinstance(v, np.ndarray):
                    result_json[k] = v.tolist()
                else:
                    try:
                        result_json[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                    except (TypeError, ValueError):
                        result_json[k] = v
            elif isinstance(v, (np.floating, np.integer)):
                result_json[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                result_json[k] = v
        
        output_file = output_dir / f"{key}.json"
        with open(output_file, 'w') as f:
            json.dump(result_json, f, indent=2)
    
    # Also save summary
    summary_file = output_dir / f"summary_{timestamp}.json"
    summary = {
        'timestamp': timestamp,
        'results': {k: {'asr': v.get('asr', 0), 'ssim': v.get('ssim', 0), 'psnr': v.get('psnr', 0)} 
                   for k, v in all_results.items()}
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive paper evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot transfer evaluation
  python experiments/run_paper_evaluation.py \\
      --agent results/agent/final_agent \\
      --victim-model checkpoints/victim/resnet18_pathmnist_best.pth \\
      --datasets pathmnist dermamnist bloodmnist \\
      --n-samples 500 --include-autoattack

  # Single dataset evaluation
  python experiments/run_paper_evaluation.py \\
      --agent results/agent/final_agent \\
      --victim-model checkpoints/victim/resnet18_pathmnist_best.pth \\
      --datasets pathmnist \\
      --n-samples 1000
        """
    )
    
    parser.add_argument('--agent', type=str, required=True,
                       help='Path to trained MedSecure agent')
    parser.add_argument('--victim-model', type=str, default=None,
                       help='Path to single victim model (for zero-shot transfer on all datasets)')
    parser.add_argument('--victim-models-dir', type=str, default='checkpoints/victim',
                       help='Directory with per-dataset models (resnet18_{dataset}_best.pth)')
    parser.add_argument('--datasets', nargs='+', 
                       default=['pathmnist', 'dermamnist', 'bloodmnist'],
                       help='Datasets to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/paper_evaluation',
                       help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='Number of samples per dataset')
    parser.add_argument('--baseline-epsilons', nargs='+', type=float,
                       default=[0.005, 0.01, 0.03],
                       help='Epsilon values for baseline methods')
    parser.add_argument('--medsecure-epsilon', type=float, default=0.03,
                       help='Maximum epsilon budget for MedSecure (agent chooses optimal)')
    parser.add_argument('--include-autoattack', action='store_true',
                       help='Include AutoAttack in evaluation (slow)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--strategies', nargs='+', default=['pixel', 'frequency'],
                       help='Attack strategies for MedSecure')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Baseline epsilons: {args.baseline_epsilons}")
    print(f"MedSecure budget: {args.medsecure_epsilon}")
    print(f"Include AutoAttack: {args.include_autoattack}")
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =================================================================
    # EVALUATE EACH DATASET
    # =================================================================
    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        # Determine victim model path
        # Priority: 1) per-dataset model in victim-models-dir, 2) single --victim-model
        victim_path = None
        
        # Try per-dataset model first
        if args.victim_models_dir:
            per_dataset_path = Path(args.victim_models_dir) / f"resnet18_{dataset}_best.pth"
            if per_dataset_path.exists():
                victim_path = str(per_dataset_path)
                print(f"Using per-dataset model: {victim_path}")
        
        # Fallback to single model
        if victim_path is None and args.victim_model:
            victim_path = args.victim_model
            print(f"Using single model (zero-shot): {victim_path}")
        
        if victim_path is None or not Path(victim_path).exists():
            print(f"ERROR: No victim model found for {dataset}")
            print(f"  Tried: {Path(args.victim_models_dir) / f'resnet18_{dataset}_best.pth'}")
            if args.victim_model:
                print(f"  Tried: {args.victim_model}")
            continue
        
        # Create evaluator
        try:
            evaluator = Evaluator(
                victim_model_path=victim_path,
                agent_path=args.agent,
                dataset_name=dataset,
                device=torch.device(device),
                output_dir=str(output_dir / dataset),
                algorithm='sac',
                strategies=args.strategies
            )
        except Exception as e:
            print(f"ERROR creating evaluator for {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Run evaluation
        try:
            dataset_results = run_full_evaluation(
                evaluator=evaluator,
                n_samples=args.n_samples,
                baseline_epsilons=args.baseline_epsilons,
                medsecure_epsilon=args.medsecure_epsilon,
                include_autoattack=args.include_autoattack
            )
            
            # Store with dataset prefix
            for key, value in dataset_results.items():
                all_results[f"{key}_{dataset}"] = value
            
            # Save results incrementally after each dataset
            _save_results_incrementally(all_results, output_dir, timestamp)
            
        except Exception as e:
            print(f"ERROR evaluating {dataset}: {e}")
            import traceback
            traceback.print_exc()
            # Still save what we have so far
            if all_results:
                _save_results_incrementally(all_results, output_dir, timestamp)
            continue
    
    # =================================================================
    # SAVE RESULTS
    # =================================================================
    
    # Save individual results per method/dataset
    for key, result in all_results.items():
        # Convert numpy types to Python types for JSON
        result_json = {}
        for k, v in result.items():
            if isinstance(v, (np.ndarray, list)):
                if isinstance(v, np.ndarray):
                    result_json[k] = v.tolist()
                else:
                    result_json[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                result_json[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                result_json[k] = v
        
        output_file = output_dir / f"{key}.json"
        with open(output_file, 'w') as f:
            json.dump(result_json, f, indent=2)
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'config': {
            'datasets': args.datasets,
            'n_samples': args.n_samples,
            'baseline_epsilons': args.baseline_epsilons,
            'medsecure_epsilon': args.medsecure_epsilon,
            'victim_model': args.victim_model,  # None if using per-dataset models
            'victim_models_dir': args.victim_models_dir,
            'agent': args.agent,
            'include_autoattack': args.include_autoattack,
        },
        'results': {}
    }
    
    for key, result in all_results.items():
        summary['results'][key] = {
            'method': result.get('method', ''),
            'asr': float(result.get('asr', 0)),
            'ssim': float(result.get('ssim', 0)),
            'psnr': float(result.get('psnr', 0)),
            'epsilon': float(result.get('epsilon', 0)),
            'epsilon_std': float(result.get('epsilon_std', 0)),
        }
    
    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # =================================================================
    # PRINT SUMMARY TABLE
    # =================================================================
    print("\n" + "="*90)
    print("EVALUATION SUMMARY")
    print("="*90)
    
    # Group by dataset
    for dataset in args.datasets:
        print(f"\n{dataset.upper()}")
        print("-"*90)
        print(f"{'Method':<20} {'Epsilon':>10} {'ASR':>10} {'SSIM':>10} {'PSNR':>10} {'Queries':>10}")
        print("-"*90)
        
        # Baselines
        for eps in args.baseline_epsilons:
            for baseline in ['fgsm', 'pgd', 'freq'] + (['autoattack'] if args.include_autoattack else []):
                key = f"{baseline}_eps{eps:.3f}_{dataset}"
                if key in all_results:
                    r = all_results[key]
                    print(f"{baseline.upper():<20} {eps:>10.3f} {r['asr']*100:>9.1f}% "
                          f"{r['ssim']:>10.4f} {r['psnr']:>10.2f} {r['queries']:>10.1f}")
        
        # MedSecure
        key = f"medsecure_{dataset}"
        if key in all_results:
            r = all_results[key]
            eps_str = f"{r['epsilon']:.3f}±{r['epsilon_std']:.3f}"
            print(f"{'MedSecure':<20} {eps_str:>10} {r['asr']*100:>9.1f}% "
                  f"{r['ssim']:>10.4f} {r['psnr']:>10.2f} {r['queries']:>10.1f}")
    
    print("\n" + "="*90)
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print("="*90)
    
    # =================================================================
    # GENERATE LATEX TABLE
    # =================================================================
    latex_file = output_dir / f"table_main_results_{timestamp}.tex"
    generate_latex_table(all_results, args.datasets, args.baseline_epsilons, 
                        args.include_autoattack, latex_file)
    print(f"LaTeX table: {latex_file}")


def generate_latex_table(
    results: Dict[str, Any],
    datasets: List[str],
    baseline_epsilons: List[float],
    include_autoattack: bool,
    output_path: Path
):
    """Generate LaTeX table from results."""
    
    baselines = ['fgsm', 'pgd', 'freq']
    if include_autoattack:
        baselines.append('autoattack')
    
    # Use middle epsilon for main table (0.02 if available)
    main_eps = 0.02 if 0.02 in baseline_epsilons else baseline_epsilons[len(baseline_epsilons)//2]
    
    latex = r"""\begin{table*}[t]
\centering
\caption{Comparison of MedSecure with baseline attacks across medical imaging datasets. 
MedSecure is trained \textbf{only on PathMNIST} and evaluated zero-shot on other datasets. 
Baselines use fixed $\epsilon=%.3f$; MedSecure adaptively selects $\epsilon$ per image.
Best results in \textbf{bold}.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" % main_eps
    
    # Column spec
    for _ in datasets:
        latex += r"ccc"
    latex += r"""}
\toprule
"""
    
    # Dataset headers
    latex += "Method"
    for dataset in datasets:
        name = dataset.replace('mnist', 'MNIST').replace('path', 'Path').replace('derma', 'Derma').replace('blood', 'Blood')
        latex += f" & \\multicolumn{{3}}{{c}}{{{name}}}"
    latex += r" \\" + "\n"
    
    # Metric headers
    latex += ""
    for _ in datasets:
        latex += r" & ASR$\uparrow$ & SSIM$\uparrow$ & $\bar{\epsilon}$"
    latex += r" \\" + "\n"
    latex += r"\midrule" + "\n"
    
    # Find best values per dataset for highlighting
    best_asr = {d: 0 for d in datasets}
    best_ssim = {d: 0 for d in datasets}
    
    for dataset in datasets:
        for baseline in baselines:
            key = f"{baseline}_eps{main_eps:.3f}_{dataset}"
            if key in results:
                best_asr[dataset] = max(best_asr[dataset], results[key]['asr'])
                best_ssim[dataset] = max(best_ssim[dataset], results[key]['ssim'])
        
        key = f"medsecure_{dataset}"
        if key in results:
            best_asr[dataset] = max(best_asr[dataset], results[key]['asr'])
            best_ssim[dataset] = max(best_ssim[dataset], results[key]['ssim'])
    
    # Baseline rows
    for baseline in baselines:
        latex += baseline.upper()
        
        for dataset in datasets:
            key = f"{baseline}_eps{main_eps:.3f}_{dataset}"
            if key in results:
                r = results[key]
                asr_str = f"{r['asr']*100:.1f}\\%"
                ssim_str = f"{r['ssim']:.3f}"
                
                # Bold if best
                if abs(r['asr'] - best_asr[dataset]) < 0.005:
                    asr_str = f"\\textbf{{{asr_str}}}"
                if abs(r['ssim'] - best_ssim[dataset]) < 0.005:
                    ssim_str = f"\\textbf{{{ssim_str}}}"
                
                latex += f" & {asr_str} & {ssim_str} & {main_eps:.3f}"
            else:
                latex += " & -- & -- & --"
        
        latex += r" \\" + "\n"
    
    latex += r"\midrule" + "\n"
    
    # MedSecure row
    latex += "\\textbf{MedSecure}"
    
    for dataset in datasets:
        key = f"medsecure_{dataset}"
        if key in results:
            r = results[key]
            asr_str = f"{r['asr']*100:.1f}\\%"
            ssim_str = f"{r['ssim']:.3f}"
            eps_str = f"{r['epsilon']:.3f}"
            if r['epsilon_std'] > 0.001:
                eps_str = f"{r['epsilon']:.2f}$\\pm${r['epsilon_std']:.2f}"
            
            # Bold if best
            if abs(r['asr'] - best_asr[dataset]) < 0.005:
                asr_str = f"\\textbf{{{asr_str}}}"
            if abs(r['ssim'] - best_ssim[dataset]) < 0.005:
                ssim_str = f"\\textbf{{{ssim_str}}}"
            
            latex += f" & {asr_str} & {ssim_str} & {eps_str}"
        else:
            latex += " & -- & -- & --"
    
    latex += r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)


if __name__ == '__main__':
    main()