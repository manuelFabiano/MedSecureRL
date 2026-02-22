#!/usr/bin/env python3
"""
Exhaustive grid search for pixel attack parameters.
Compares optimal parameters found by brute-force with MedSecure's adaptive selection.
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from attacks.pixel_attack import PixelAttack
from attacks import ATTACK_REGISTRY
from rl.hierarchical_env import HierarchicalAttackEnv
from rl.hierarchical_agent_sac import create_hierarchical_agent_sac


def compute_ssim(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute SSIM between original and adversarial images."""
    try:
        from skimage.metrics import structural_similarity
        orig_np = original.squeeze().cpu().detach().numpy().clip(0, 1)
        adv_np = adversarial.squeeze().cpu().detach().numpy().clip(0, 1)
        
        if orig_np.ndim == 3:
            orig_np = orig_np.transpose(1, 2, 0)
            adv_np = adv_np.transpose(1, 2, 0)
        
        return structural_similarity(
            orig_np, adv_np, 
            data_range=1.0,
            channel_axis=-1 if orig_np.ndim == 3 else None
        )
    except:
        return 0.5


def compute_psnr(original: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute PSNR between original and adversarial images."""
    try:
        mse = ((original - adversarial) ** 2).mean().item()
        if mse > 1e-10:
            return 10 * np.log10(1.0 / mse)
        return 100.0
    except:
        return 30.0


@dataclass
class AttackResult:
    """Result of a single attack configuration."""
    epsilon: float
    iterations: int
    step_size: float
    success: bool
    ssim: float
    psnr: float
    queries: int


def load_victim_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load victim model and normalization stats."""
    from models.victim_models import get_victim_model
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    num_classes = checkpoint.get('n_classes', checkpoint.get('num_classes', 9))
    model_name = checkpoint.get('model_name', 'resnet18')
    normalize_stats = checkpoint.get('normalize_stats', None)
    
    # Load model using the standard function
    model = get_victim_model(
        model_name,
        num_classes=num_classes,
        checkpoint=model_path,
        device=str(device)
    )
    model.eval()
    
    return model, normalize_stats


def load_dataset(dataset_name: str, n_samples: int, device: torch.device, normalize_stats: Dict) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Load correctly classified samples from dataset."""
    import medmnist
    from medmnist import INFO
    from torchvision import transforms
    
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Build transform
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    ]
    if normalize_stats:
        transform_list.append(
            transforms.Normalize(normalize_stats['mean'], normalize_stats['std'])
        )
    transform = transforms.Compose(transform_list)
    
    dataset = DataClass(split='test', transform=transform, download=True)
    
    return dataset


def select_hard_samples(
    model: nn.Module,
    dataset,
    n_hard_samples: int,
    device: torch.device,
    normalize_stats: Optional[Dict],
    test_epsilon: float = 0.03,
    test_iterations: int = 40,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Select hard-to-attack samples by running PGD and keeping failures.
    
    Returns:
        List of (image, label, original_confidence) tuples for samples
        where PGD failed to cause misclassification.
    """
    from tqdm import tqdm
    
    attack = PixelAttack(
        model=model,
        epsilon=test_epsilon,
        method='pgd',
        iterations=test_iterations,
        device=str(device),
        normalize_stats=normalize_stats,
    )
    
    hard_samples = []
    total_tested = 0
    
    print(f"\n  Scanning for hard samples (PGD with ε={test_epsilon}, {test_iterations} iters)...")
    
    for i, (img, lbl) in enumerate(tqdm(dataset, desc="  Scanning")):
        img = img.unsqueeze(0).to(device)
        lbl_val = lbl.item() if hasattr(lbl, 'item') else int(lbl)
        lbl = torch.tensor([lbl_val]).to(device)
        
        # Check if correctly classified
        with torch.no_grad():
            pred = model(img).argmax(dim=1)
        if pred.item() != lbl.item():
            continue  # Skip misclassified samples
        
        total_tested += 1
        
        # Get original confidence
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(img), dim=-1)
            orig_conf = probs[0, lbl.item()].item()
        
        # Run PGD attack
        result = attack.attack(x=img, y=lbl, epsilon=test_epsilon, iterations=test_iterations)
        
        # Check if attack FAILED (this is a hard sample)
        if not result.success:
            hard_samples.append((img, lbl, orig_conf))
            print(f"    Found hard sample #{len(hard_samples)}: idx={i}, conf={orig_conf:.4f}")
            
            if len(hard_samples) >= n_hard_samples:
                break
        
        # Stop after checking enough samples
        if total_tested >= 1000:
            break
    
    print(f"\n  Scanned {total_tested} correctly classified samples")
    print(f"  Found {len(hard_samples)} hard samples (PGD failed)")
    if total_tested > 0:
        print(f"  Attack success rate on scanned: {100*(total_tested - len(hard_samples))/total_tested:.1f}%")
    
    if len(hard_samples) < n_hard_samples:
        print(f"  WARNING: Only found {len(hard_samples)} hard samples, wanted {n_hard_samples}")
    
    return hard_samples


def grid_search_single_image(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    epsilon_values: np.ndarray,
    iteration_values: np.ndarray,
    step_size_values: np.ndarray,
    normalize_stats: Optional[Dict],
    device: torch.device,
) -> Tuple[Optional[AttackResult], int, int, float]:
    """
    Exhaustive grid search for a single image.
    
    Returns:
        best_result: Best attack result (min epsilon among successes, or None if no success)
        total_combinations: Total number of combinations tried
        total_queries: Total number of model forward passes
        elapsed_time: Time taken in seconds
    """
    attack = PixelAttack(
        model=model,
        epsilon=0.03,  # Will be overridden
        method='pgd',
        device=str(device),
        normalize_stats=normalize_stats,
    )
    
    best_result = None
    total_combinations = 0
    total_queries = 0  # Count all forward passes
    
    # Check original prediction is correct
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=-1).item()
        true_label = label.item()
    
    # CRITICAL: Verify original prediction is correct
    if pred_class != true_label:
        return None, 0, 0, 0.0
    
    start_time = time.time()
    
    # Try all combinations
    for eps in epsilon_values:
        for iters in iteration_values:
            for step in step_size_values:
                total_combinations += 1
                
                # Run attack (random_start=False for deterministic test)
                result = attack.attack(
                    x=image,
                    y=label,
                    epsilon=eps,
                    iterations=int(iters),
                    step_size=step,
                    random_start=False,  # BIM-style, more deterministic
                )
                
                # Accumulate queries (forward passes)
                total_queries += result.queries
                
                # Compute metrics
                if normalize_stats:
                    mean = torch.tensor(normalize_stats['mean']).view(1, -1, 1, 1).to(device)
                    std = torch.tensor(normalize_stats['std']).view(1, -1, 1, 1).to(device)
                    orig_denorm = image * std + mean
                    adv_denorm = result.adversarial * std + mean
                else:
                    orig_denorm = image
                    adv_denorm = result.adversarial
                
                ssim = compute_ssim(orig_denorm, adv_denorm)
                psnr = compute_psnr(orig_denorm, adv_denorm)
                
                attack_result = AttackResult(
                    epsilon=eps,
                    iterations=int(iters),
                    step_size=step,
                    success=result.success,
                    ssim=ssim,
                    psnr=psnr,
                    queries=result.queries,
                )
                
                # Update best if this is successful with acceptable quality
                # Criterion: minimum epsilon among successes with SSIM >= 0.85
                if result.success and ssim >= 0.85:
                    if best_result is None:
                        best_result = attack_result
                    elif eps < best_result.epsilon:
                        # Lower epsilon is better (epsilon efficiency)
                        best_result = attack_result
                    elif eps == best_result.epsilon and ssim > best_result.ssim:
                        # Same epsilon, higher SSIM is better
                        best_result = attack_result
    
    elapsed_time = time.time() - start_time
    
    return best_result, total_combinations, total_queries, elapsed_time


def run_medsecure_single_image(
    agent,
    env,
    image: torch.Tensor,
    label: torch.Tensor,
    normalize_stats: Optional[Dict],
    device: torch.device,
) -> Tuple[Optional[AttackResult], float]:
    """
    Run MedSecure on a single image.
    
    Returns:
        result: Attack result
        elapsed_time: Time taken in seconds
    """
    start_time = time.time()
    
    # Reset environment with specific image
    obs, info = env.reset(options={'image': image, 'label': label})
    
    original = env.current_image.clone()
    
    # Get base state (21 dims) - controllers expect this, not full obs (23 dims)
    base_state = obs[:21]
    
    # Meta-controller selects strategy (force PIXEL for fair comparison)
    env.set_strategy(0)  # PIXEL = 0
    
    # SAC selects parameters using base_state (21 dims), NOT full obs
    action, _ = agent.controllers[0].predict(base_state, deterministic=True)
    
    # Execute attack
    obs, reward, terminated, truncated, info = env.step(action)
    
    elapsed_time = time.time() - start_time
    
    # Get results
    adversarial = env.current_adversarial
    if adversarial is None:
        adversarial = original
    
    # Compute metrics
    if normalize_stats:
        mean = torch.tensor(normalize_stats['mean']).view(1, -1, 1, 1).to(device)
        std = torch.tensor(normalize_stats['std']).view(1, -1, 1, 1).to(device)
        orig_denorm = original * std + mean
        adv_denorm = adversarial * std + mean
    else:
        orig_denorm = original
        adv_denorm = adversarial
    
    ssim = compute_ssim(orig_denorm, adv_denorm)
    psnr = compute_psnr(orig_denorm, adv_denorm)
    
    params = info.get('params', {})
    
    result = AttackResult(
        epsilon=params.get('epsilon', 0),
        iterations=params.get('iterations', 0),
        step_size=params.get('step_size', 0),
        success=info.get('success', False),
        ssim=ssim,
        psnr=psnr,
        queries=info.get('queries', 0),
    )
    
    return result, elapsed_time


def main():
    parser = argparse.ArgumentParser(description='Exhaustive grid search vs MedSecure')
    parser.add_argument('--victim-model', type=str, required=True,
                        help='Path to victim model')
    parser.add_argument('--agent', type=str, required=True,
                        help='Path to MedSecure agent')
    parser.add_argument('--dataset', type=str, default='pathmnist',
                        help='Dataset name')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of images to test')
    parser.add_argument('--output-dir', type=str, default='results/grid_search',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    
    # Grid search parameters
    parser.add_argument('--eps-min', type=float, default=0.001)
    parser.add_argument('--eps-max', type=float, default=0.03)
    parser.add_argument('--eps-steps', type=int, default=30)
    parser.add_argument('--iter-min', type=int, default=5)
    parser.add_argument('--iter-max', type=int, default=50)
    parser.add_argument('--iter-step', type=int, default=2)
    parser.add_argument('--step-min', type=float, default=0.001)
    parser.add_argument('--step-max', type=float, default=0.01)
    parser.add_argument('--step-steps', type=int, default=20)
    parser.add_argument('--hard-samples', action='store_true',
                        help='Select hard-to-attack samples (where PGD fails)')
    parser.add_argument('--hard-eps', type=float, default=0.03,
                        help='Epsilon for hard sample selection')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load victim model
    print(f"Loading victim model from {args.victim_model}")
    model, normalize_stats = load_victim_model(args.victim_model, device)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}")
    dataset = load_dataset(args.dataset, args.n_samples, device, normalize_stats)
    
    # Create parameter grids
    epsilon_values = np.linspace(args.eps_min, args.eps_max, args.eps_steps)
    iteration_values = np.arange(args.iter_min, args.iter_max, args.iter_step)
    step_size_values = np.linspace(args.step_min, args.step_max, args.step_steps)
    
    total_combinations = len(epsilon_values) * len(iteration_values) * len(step_size_values)
    print(f"\nGrid search configuration:")
    print(f"  Epsilon: {len(epsilon_values)} values in [{args.eps_min}, {args.eps_max}]")
    print(f"  Iterations: {len(iteration_values)} values in [{args.iter_min}, {args.iter_max})")
    print(f"  Step size: {len(step_size_values)} values in [{args.step_min}, {args.step_max}]")
    print(f"  Total combinations per image: {total_combinations:,}")
    print(f"  Optimality criterion: min epsilon with success and SSIM >= 0.85")
    
    # Load MedSecure agent
    print(f"\nLoading MedSecure agent from {args.agent}")
    
    # Create a minimal dataloader for env initialization
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Environment config - MUST match training range [0.001, target_epsilon]
    env_config = {
        "max_steps_per_image": 5,
        "rl": {
            "action_space": {
                "epsilon_range": [0.001, args.eps_max],  # Same as training
                "iterations_range": [args.iter_min, args.iter_max],
                "step_size_range": [args.step_min, args.step_max]
            }
        },
        "reward": {
            "weights": {"success": 1.0, "imperceptibility": 0.3, "efficiency": 0.05}
        }
    }
    
    env = HierarchicalAttackEnv(
        victim_model=model,
        attack_registry=ATTACK_REGISTRY,
        dataloader=train_loader,
        config=env_config,
        device=str(device),
        mode="separated",
        allowed_strategies=['pixel', 'frequency'],  # Must match training config
        normalize_stats=normalize_stats
    )
    print(f"  MedSecure epsilon range: [0.001, {args.eps_max}]")
    print(f"  Grid search epsilon range: [{args.eps_min}, {args.eps_max}]")
    
    # Load state extractor statistics if available
    agent_path = Path(args.agent)
    state_stats_path = agent_path / "state_extractor_stats.pt"
    if state_stats_path.exists():
        env.state_extractor.load_stats(str(state_stats_path))
        env.state_extractor.eval()
        print(f"  Loaded state extractor stats")
    
    # Agent config
    agent_config = {
        "sac_lr": 3e-4,
        "sac_buffer_size": 100000,
        "sac_batch_size": 256,
        "sac_learning_starts": 1000,
        "meta_lr": 1e-4,
        "meta_buffer_size": 10000,
        "meta_batch_size": 32,
        "meta_epsilon_start": 0.0,
        "meta_epsilon_min": 0.0,
        "meta_epsilon_decay": 1.0,
    }
    
    # Create and load agent
    agent = create_hierarchical_agent_sac(env, agent_config, str(device))
    agent.load(str(agent_path))
    
    # Select samples
    if args.hard_samples:
        # Select hard-to-attack samples (where PGD fails)
        print(f"\nSelecting {args.n_samples} HARD samples (PGD-resistant)...")
        hard_samples = select_hard_samples(
            model=model,
            dataset=dataset,
            n_hard_samples=args.n_samples,
            device=device,
            normalize_stats=normalize_stats,
            test_epsilon=args.hard_eps,
            test_iterations=40,
        )
        samples = [(img, lbl) for img, lbl, conf in hard_samples]
    else:
        # Select any correctly classified samples
        print(f"\nSelecting {args.n_samples} correctly classified samples...")
        samples = []
        with torch.no_grad():
            for i, (img, lbl) in enumerate(dataset):
                img = img.unsqueeze(0).to(device)
                lbl = torch.tensor([lbl.item()]).to(device)
                
                pred = model(img).argmax(dim=1)
                if pred.item() == lbl.item():
                    samples.append((img, lbl))
                    if len(samples) >= args.n_samples:
                        break
    
    print(f"  Selected {len(samples)} samples")
    
    # Run comparison
    results = []
    
    print(f"\n{'='*80}")
    print("Running comparison...")
    print(f"{'='*80}\n")
    
    for idx, (image, label) in enumerate(samples):
        print(f"\n--- Image {idx+1}/{len(samples)} ---")
        
        # Grid search
        print(f"  Running grid search ({total_combinations:,} combinations)...")
        grid_result, n_combinations, grid_queries, grid_time = grid_search_single_image(
            model=model,
            image=image,
            label=label,
            epsilon_values=epsilon_values,
            iteration_values=iteration_values,
            step_size_values=step_size_values,
            normalize_stats=normalize_stats,
            device=device,
        )
        
        # MedSecure
        print(f"  Running MedSecure...")
        medsecure_result, medsecure_time = run_medsecure_single_image(
            agent=agent,
            env=env,
            image=image,
            label=label,
            normalize_stats=normalize_stats,
            device=device,
        )
        
        # Store results
        result = {
            'image_idx': idx,
            'grid_search': {
                'success': grid_result.success if grid_result else False,
                'epsilon': grid_result.epsilon if grid_result else None,
                'iterations': grid_result.iterations if grid_result else None,
                'step_size': grid_result.step_size if grid_result else None,
                'ssim': grid_result.ssim if grid_result else None,
                'psnr': grid_result.psnr if grid_result else None,
                'queries': grid_queries,  # Total queries for this image
                'best_queries': grid_result.queries if grid_result else None,  # Queries of best config
                'time': grid_time,
                'combinations': n_combinations,
            },
            'medsecure': {
                'success': medsecure_result.success if medsecure_result else False,
                'epsilon': medsecure_result.epsilon if medsecure_result else None,
                'iterations': medsecure_result.iterations if medsecure_result else None,
                'step_size': medsecure_result.step_size if medsecure_result else None,
                'ssim': medsecure_result.ssim if medsecure_result else None,
                'psnr': medsecure_result.psnr if medsecure_result else None,
                'queries': medsecure_result.queries if medsecure_result else None,
                'time': medsecure_time,
            },
        }
        results.append(result)
        
        # Print comparison
        print(f"\n  {'Method':<15} {'Success':<10} {'Epsilon':<10} {'Iters':<8} {'Step':<10} {'SSIM':<8} {'PSNR':<8} {'Queries':<12} {'Time':<10}")
        print(f"  {'-'*95}")
        
        if grid_result:
            print(f"  {'Grid Search':<15} {str(grid_result.success):<10} {grid_result.epsilon:<10.4f} {grid_result.iterations:<8} {grid_result.step_size:<10.4f} {grid_result.ssim:<8.4f} {grid_result.psnr:<8.2f} {grid_queries:<12,} {grid_time:<10.2f}s")
        else:
            print(f"  {'Grid Search':<15} {'False':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'N/A':<8} {grid_queries:<12,} {grid_time:<10.2f}s")
        
        if medsecure_result:
            print(f"  {'MedSecure':<15} {str(medsecure_result.success):<10} {medsecure_result.epsilon:<10.4f} {medsecure_result.iterations:<8} {medsecure_result.step_size:<10.4f} {medsecure_result.ssim:<8.4f} {medsecure_result.psnr:<8.2f} {medsecure_result.queries:<12} {medsecure_time:<10.4f}s")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    grid_successes = sum(1 for r in results if r['grid_search']['success'])
    medsecure_successes = sum(1 for r in results if r['medsecure']['success'])
    
    grid_ssims = [r['grid_search']['ssim'] for r in results if r['grid_search']['success'] and r['grid_search']['ssim']]
    medsecure_ssims = [r['medsecure']['ssim'] for r in results if r['medsecure']['success'] and r['medsecure']['ssim']]
    
    grid_times = [r['grid_search']['time'] for r in results]
    medsecure_times = [r['medsecure']['time'] for r in results]
    
    grid_queries_total = [r['grid_search']['queries'] for r in results]
    medsecure_queries_total = [r['medsecure']['queries'] for r in results if r['medsecure']['queries']]
    
    print(f"\n  Success Rate:")
    print(f"    Grid Search: {grid_successes}/{len(results)} ({100*grid_successes/len(results):.1f}%)")
    print(f"    MedSecure:   {medsecure_successes}/{len(results)} ({100*medsecure_successes/len(results):.1f}%)")
    
    if grid_ssims and medsecure_ssims:
        print(f"\n  Mean SSIM (successful attacks):")
        print(f"    Grid Search: {np.mean(grid_ssims):.4f}")
        print(f"    MedSecure:   {np.mean(medsecure_ssims):.4f}")
    
    print(f"\n  Mean Queries per Image:")
    print(f"    Grid Search: {np.mean(grid_queries_total):,.0f}")
    print(f"    MedSecure:   {np.mean(medsecure_queries_total):.1f}")
    print(f"    Speedup:     {np.mean(grid_queries_total) / np.mean(medsecure_queries_total):,.0f}x")
    
    print(f"\n  Mean Time per Image:")
    print(f"    Grid Search: {np.mean(grid_times):.2f}s")
    print(f"    MedSecure:   {np.mean(medsecure_times):.4f}s")
    print(f"    Speedup:     {np.mean(grid_times) / np.mean(medsecure_times):.0f}x")
    
    # Parameter comparison
    print(f"\n  Parameter Comparison (successful attacks on same images):")
    both_success = [(r['grid_search'], r['medsecure']) for r in results 
                    if r['grid_search']['success'] and r['medsecure']['success']]
    
    if both_success:
        eps_diffs = [abs(g['epsilon'] - m['epsilon']) for g, m in both_success]
        iter_diffs = [abs(g['iterations'] - m['iterations']) for g, m in both_success]
        step_diffs = [abs(g['step_size'] - m['step_size']) for g, m in both_success]
        ssim_diffs = [g['ssim'] - m['ssim'] for g, m in both_success]  # Positive = grid better
        
        print(f"    Mean |Δε|:         {np.mean(eps_diffs):.4f}")
        print(f"    Mean |Δiters|:     {np.mean(iter_diffs):.1f}")
        print(f"    Mean |Δstep_size|: {np.mean(step_diffs):.4f}")
        print(f"    Mean ΔSSIM:        {np.mean(ssim_diffs):+.4f} (positive = grid better)")
    
    # Save results
    output_file = output_dir / 'grid_search_comparison.json'
    
    # Helper to convert numpy types to Python native types
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(to_serializable({
            'config': {
                'n_samples': len(samples),
                'epsilon_values': epsilon_values.tolist(),
                'iteration_values': iteration_values.tolist(),
                'step_size_values': step_size_values.tolist(),
                'total_combinations': total_combinations,
            },
            'results': results,
            'summary': {
                'grid_search': {
                    'success_rate': grid_successes / len(results),
                    'mean_ssim': float(np.mean(grid_ssims)) if grid_ssims else None,
                    'mean_queries': float(np.mean(grid_queries_total)),
                    'mean_time': float(np.mean(grid_times)),
                },
                'medsecure': {
                    'success_rate': medsecure_successes / len(results),
                    'mean_ssim': float(np.mean(medsecure_ssims)) if medsecure_ssims else None,
                    'mean_queries': float(np.mean(medsecure_queries_total)) if medsecure_queries_total else None,
                    'mean_time': float(np.mean(medsecure_times)),
                },
                'query_speedup': float(np.mean(grid_queries_total) / np.mean(medsecure_queries_total)) if medsecure_queries_total else 0,
                'time_speedup': float(np.mean(grid_times) / np.mean(medsecure_times)) if np.mean(medsecure_times) > 0 else 0,
            }
        }), f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")


if __name__ == '__main__':
    main()