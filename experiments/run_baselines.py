"""
Run baseline adversarial attacks for comparison.

Executes standard attack methods (FGSM, PGD, BIM, C&W, AutoAttack)
and records performance metrics for comparison with the RL agent.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloader, get_dataset
from models import get_victim_model
from attacks import get_attack
from attacks.baselines import FGSM, PGD, CarliniWagner, AutoAttackWrapper
from evaluation import ComprehensiveMetrics, ComprehensiveEfficiencyMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineRunner:
    """
    Runner for baseline adversarial attacks.
    """
    
    BASELINE_ATTACKS = {
        'fgsm': {'class': FGSM, 'params': {'epsilon': 0.03}},
        'pgd': {'class': PGD, 'params': {'epsilon': 0.03, 'iterations': 40, 'step_size': 0.01}},
        'bim': {'class': PGD, 'params': {'epsilon': 0.03, 'iterations': 40, 'step_size': 0.01, 'random_start': False}},
        'cw': {'class': CarliniWagner, 'params': {'initial_c': 0.001, 'confidence': 0.0, 'max_iterations': 1000, 'learning_rate': 0.01}},
    }
    
    def __init__(
        self,
        model_path: str,
        dataset_name: str = 'pathmnist',
        device: Optional[torch.device] = None,
        output_dir: str = 'results/baselines'
    ):
        """
        Args:
            model_path: Path to victim model checkpoint
            dataset_name: Dataset name
            device: Device for computation
            output_dir: Directory for saving results
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load victim model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = get_victim_model(
            checkpoint['model_name'],
            num_classes=checkpoint['n_classes'],
            checkpoint=model_path  # This will auto-detect native/small_input
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.n_classes = checkpoint['n_classes']
        
        # Get native flag from checkpoint
        self.native = checkpoint.get('native', False)
        
        # Get normalization stats from checkpoint
        if self.native:
            # Default for native mode
            n_channels = 3 if 'path' in dataset_name or 'derma' in dataset_name or 'blood' in dataset_name else 1
            default_stats = {'mean': [0.5]*n_channels, 'std': [0.5]*n_channels}
        else:
            # Default ImageNet for upscaled
            default_stats = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        self.normalize_stats = checkpoint.get('normalize_stats', default_stats)
        
        logger.info(f"Loaded victim model: {checkpoint['model_name']}")
        logger.info(f"Model accuracy: {checkpoint.get('accuracy', 'unknown')}")
        logger.info(f"Native mode: {self.native} ({'28x28' if self.native else '224x224'})")
        logger.info(f"Normalization stats: mean={self.normalize_stats['mean']}, std={self.normalize_stats['std']}")
    
    def verify_accuracy(self, n_samples: int = None, batch_size: int = 32) -> float:
        """
        Verify model test accuracy.
        
        Args:
            n_samples: Number of samples to evaluate (None = full test set)
            batch_size: Batch size
            
        Returns:
            Test accuracy as float
        """
        dataset = get_dataset(self.dataset_name, split='test', native=self.native)
        loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Verifying accuracy"):
                if n_samples is not None and total >= n_samples:
                    break
                    
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        logger.info(f"Test Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy
    
    def _get_normalize_stats(self) -> Dict[str, List[float]]:
        """Get normalization statistics from the loaded checkpoint.
        
        Returns:
            Dictionary with 'mean' and 'std' lists
        """
        return self.normalize_stats
    
    def run_attack(
        self,
        attack_name: str,
        n_samples: int = 1000,
        batch_size: int = 32,
        epsilon: Optional[float] = None,
        **attack_kwargs
    ) -> Dict[str, Any]:
        """
        Run a single baseline attack.
        
        Args:
            attack_name: Name of attack ('fgsm', 'pgd', 'bim', 'cw', 'autoattack')
            n_samples: Number of samples to attack
            batch_size: Batch size
            epsilon: Override default epsilon
            **attack_kwargs: Additional attack parameters
        
        Returns:
            Dictionary with attack results and metrics
        """
        logger.info(f"Running {attack_name.upper()} attack...")
        
        # Get data loader with correct transforms (native flag)
        dataset = get_dataset(self.dataset_name, split='test', native=self.native)
        loader = get_dataloader(
            dataset,
            batch_size=batch_size, shuffle=False
        )
        
        # Initialize attack
        normalize_stats = self._get_normalize_stats()
        
        if attack_name == 'autoattack':
            attack = AutoAttackWrapper(
                self.model,
                norm='Linf',
                eps=epsilon or 0.03,
                device=self.device
            )
        else:
            attack_config = self.BASELINE_ATTACKS[attack_name]
            params = attack_config['params'].copy()
            if epsilon is not None:
                params['epsilon'] = epsilon
            params.update(attack_kwargs)
            
            # Create attack with normalization stats
            attack = attack_config['class'](
                model=self.model,
                device=self.device,
                normalize_stats=normalize_stats,
                **params
            )
        
        # Initialize metrics with normalization stats
        metrics = ComprehensiveMetrics(
            compute_lpips=True,
            device=self.device,
            normalize_stats=normalize_stats
        )
        efficiency = ComprehensiveEfficiencyMetrics(device=self.device)
        
        samples_processed = 0
        all_results = []
        
        for images, labels in tqdm(loader, desc=f"Running {attack_name}"):
            if samples_processed >= n_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            # Get original predictions
            with torch.no_grad():
                original_outputs = self.model(images)
                original_preds = original_outputs.argmax(dim=1)
            
            # Run attack
            efficiency.start_attack()
            
            if attack_name == 'autoattack':
                adversarial = attack.generate(images, labels)
                attack_queries = images.size(0)  # AutoAttack doesn't track queries
            else:
                result = attack.attack(images, labels)
                adversarial = result.adversarial
                attack_queries = result.queries  # Get queries from attack result
            
            # Pass queries to efficiency tracker
            efficiency.queries.count_forward(attack_queries)
            
            # Get adversarial predictions
            with torch.no_grad():
                adversarial_outputs = self.model(adversarial)
                adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            perturbation_norm = (adversarial - images).view(images.size(0), -1).norm(p=2, dim=1).mean().item()
            success = (original_preds == labels) & (adversarial_preds != labels)
            efficiency.end_attack(success.any().item(), perturbation_norm)
            
            # Update metrics
            metrics.update(
                original=images,
                adversarial=adversarial,
                original_preds=original_preds,
                adversarial_preds=adversarial_preds,
                true_labels=labels
            )
            
            samples_processed += images.size(0)
        
        # Compute final metrics
        metric_results = metrics.summary()
        efficiency_results = efficiency.summary()
        
        results = {
            'attack_name': attack_name,
            'n_samples': samples_processed,
            'metrics': metric_results,
            'efficiency': efficiency_results,
            'params': attack_config['params'] if attack_name != 'autoattack' else {'epsilon': epsilon or 0.03}
        }
        
        logger.info(f"{attack_name.upper()} Results:")
        logger.info(f"  ASR: {metric_results['ASR']:.2%}")
        logger.info(f"  SSIM: {metric_results['SSIM']:.4f}")
        logger.info(f"  PSNR: {metric_results['PSNR']:.2f} dB")
        
        return results
    
    def run_all_baselines(
        self,
        n_samples: int = 1000,
        batch_size: int = 32,
        epsilon: float = 0.03,
        include_autoattack: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all baseline attacks.
        
        Args:
            n_samples: Number of samples per attack
            batch_size: Batch size
            epsilon: Epsilon for all attacks
            include_autoattack: Whether to include AutoAttack (slow)
        
        Returns:
            Dictionary mapping attack name to results
        """
        attacks_to_run = list(self.BASELINE_ATTACKS.keys())
        if include_autoattack:
            attacks_to_run.append('autoattack')
        
        all_results = {}
        
        for attack_name in attacks_to_run:
            try:
                results = self.run_attack(
                    attack_name,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    epsilon=epsilon
                )
                all_results[attack_name] = results
            except Exception as e:
                logger.error(f"Error running {attack_name}: {e}")
                continue
        
        # Save results
        self._save_results(all_results, epsilon)
        
        return all_results
    
    def run_epsilon_sweep(
        self,
        attack_name: str = 'pgd',
        epsilons: List[float] = [0.01, 0.02, 0.03, 0.05, 0.1],
        n_samples: int = 500,
        batch_size: int = 32
    ) -> Dict[float, Dict[str, Any]]:
        """
        Run attack across multiple epsilon values.
        
        Args:
            attack_name: Attack to sweep
            epsilons: List of epsilon values
            n_samples: Samples per epsilon
            batch_size: Batch size
        
        Returns:
            Dictionary mapping epsilon to results
        """
        sweep_results = {}
        
        for eps in epsilons:
            logger.info(f"Running {attack_name} with epsilon={eps}")
            results = self.run_attack(
                attack_name,
                n_samples=n_samples,
                batch_size=batch_size,
                epsilon=eps
            )
            sweep_results[eps] = results
        
        # Save sweep results
        sweep_path = self.output_dir / f'{attack_name}_epsilon_sweep.json'
        with open(sweep_path, 'w') as f:
            # Convert float keys to strings for JSON
            json_results = {str(k): v for k, v in sweep_results.items()}
            json.dump(json_results, f, indent=2, default=float)
        
        logger.info(f"Saved epsilon sweep results to {sweep_path}")
        
        return sweep_results
    
    def _save_results(self, results: Dict[str, Any], epsilon: float):
        """Save baseline results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'baselines_eps{epsilon}_{timestamp}.json'
        path = self.output_dir / filename
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        logger.info(f"Saved baseline results to {path}")


def run_baseline_attacks(
    model_path: str,
    dataset_name: str = 'pathmnist',
    n_samples: int = 1000,
    epsilon: float = 0.03,
    output_dir: str = 'results/baselines'
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to run all baseline attacks.
    
    Args:
        model_path: Path to victim model
        dataset_name: Dataset name
        n_samples: Number of samples
        epsilon: Perturbation budget
        output_dir: Output directory
    
    Returns:
        Results dictionary
    """
    runner = BaselineRunner(
        model_path=model_path,
        dataset_name=dataset_name,
        output_dir=output_dir
    )
    
    return runner.run_all_baselines(
        n_samples=n_samples,
        epsilon=epsilon
    )


def main():
    """Main entry point for baseline evaluation."""
    parser = argparse.ArgumentParser(description='Run baseline adversarial attacks')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to victim model checkpoint')
    parser.add_argument('--dataset', type=str, default='pathmnist',
                       help='Dataset name')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to attack')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Perturbation budget')
    parser.add_argument('--output-dir', type=str, default='results/baselines',
                       help='Output directory')
    parser.add_argument('--attack', type=str, default='all',
                       choices=['all', 'fgsm', 'pgd', 'bim', 'cw', 'autoattack'],
                       help='Attack to run')
    parser.add_argument('--include-autoattack', action='store_true',
                       help='Include AutoAttack (slow)')
    parser.add_argument('--epsilon-sweep', action='store_true',
                       help='Run epsilon sweep instead of single evaluation')
    parser.add_argument('--verify-accuracy', action='store_true',
                       help='Verify and display model test accuracy before attacks')
    
    args = parser.parse_args()
    
    runner = BaselineRunner(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )
    
    # Verify accuracy if requested
    if args.verify_accuracy:
        print("\n" + "="*60)
        print("MODEL ACCURACY VERIFICATION")
        print("="*60)
        test_acc = runner.verify_accuracy(
            n_samples=args.n_samples,
            batch_size=args.batch_size
        )
        print(f"\nTest Accuracy: {test_acc:.2%}")
        print("="*60 + "\n")
    
    if args.epsilon_sweep:
        results = runner.run_epsilon_sweep(
            attack_name=args.attack if args.attack != 'all' else 'pgd',
            n_samples=args.n_samples,
            batch_size=args.batch_size
        )
    elif args.attack == 'all':
        results = runner.run_all_baselines(
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            epsilon=args.epsilon,
            include_autoattack=args.include_autoattack
        )
    else:
        results = runner.run_attack(
            args.attack,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            epsilon=args.epsilon
        )
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE ATTACK SUMMARY")
    print("="*60)
    
    if isinstance(results, dict) and 'attack_name' not in results:
        # Multiple attacks
        for name, res in results.items():
            print(f"\n{name.upper()}:")
            print(f"  ASR: {res['metrics']['ASR']:.2%}")
            print(f"  SSIM: {res['metrics']['SSIM']:.4f}")
            print(f"  PSNR: {res['metrics']['PSNR']:.2f} dB")
            print(f"  Queries: {res['efficiency'].get('mean_queries_per_attack', 'N/A')}")
    else:
        print(f"\n{results['attack_name'].upper()}:")
        print(f"  ASR: {results['metrics']['ASR']:.2%}")
        print(f"  SSIM: {results['metrics']['SSIM']:.4f}")
        print(f"  PSNR: {results['metrics']['PSNR']:.2f} dB")


if __name__ == '__main__':
    main()