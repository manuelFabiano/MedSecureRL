"""
Ablation studies for MedSecure.

Systematically evaluates the contribution of each component:
- Attack strategies (pixel, frequency, patch, semantic)
- Reward components (success, imperceptibility, efficiency, diversity)
- Curriculum learning
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import copy

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_agent import AgentTrainer
from experiments.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AblationRunner:
    """Runner for ablation studies."""
    
    ABLATION_CONFIGS = {
        'no_frequency': {
            'description': 'Without frequency-domain attacks',
            'strategies': ['pixel', 'patch', 'semantic']
        },
        'no_patch': {
            'description': 'Without patch-based attacks',
            'strategies': ['pixel', 'frequency', 'semantic']
        },
        'no_semantic': {
            'description': 'Without semantic attacks',
            'strategies': ['pixel', 'frequency', 'patch']
        },
        'pixel_only': {
            'description': 'Only pixel-based attacks',
            'strategies': ['pixel']
        },
        'no_curriculum': {
            'description': 'Without curriculum learning',
            'curriculum_phases': [{'epsilon': 0.03, 'max_iter': 50, 'strategies': ['pixel', 'frequency', 'patch', 'semantic']}],
            'phase_length': 100000
        },
        'no_diversity_reward': {
            'description': 'Without diversity reward',
            'diversity_weight': 0.0
        },
        'no_efficiency_reward': {
            'description': 'Without efficiency reward',
            'efficiency_weight': 0.0
        },
        'full': {
            'description': 'Full MedSecure (baseline)',
            'strategies': ['pixel', 'frequency', 'patch', 'semantic']
        }
    }
    
    def __init__(
        self,
        victim_model_path: str,
        dataset_name: str = 'pathmnist',
        output_dir: str = 'results/ablation',
        device: Optional[torch.device] = None
    ):
        self.victim_model_path = victim_model_path
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results: Dict[str, Any] = {}
    
    def run_single_ablation(
        self,
        ablation_name: str,
        train_episodes: int = 20000,
        eval_samples: int = 500,
        epsilon: float = 0.03
    ) -> Dict[str, Any]:
        """Run a single ablation experiment."""
        logger.info(f"Running ablation: {ablation_name}")
        
        config = copy.deepcopy(self.ABLATION_CONFIGS.get(ablation_name, {}))
        config['epsilon'] = epsilon
        
        # Train agent with ablated config
        ablation_dir = self.output_dir / ablation_name
        trainer = AgentTrainer(
            model_path=self.victim_model_path,
            dataset_name=self.dataset_name,
            output_dir=str(ablation_dir),
            config=config,
            device=self.device
        )
        
        history = trainer.train(
            total_episodes=train_episodes,
            eval_interval=2000,
            verbose=True
        )
        
        # Evaluate
        evaluator = Evaluator(
            victim_model_path=self.victim_model_path,
            agent_path=str(ablation_dir / 'agent_best.pth'),
            dataset_name=self.dataset_name,
            output_dir=str(ablation_dir / 'eval'),
            device=self.device
        )
        
        eval_results = evaluator.evaluate_agent(n_samples=eval_samples, epsilon=epsilon)
        
        return {
            'config': config,
            'description': self.ABLATION_CONFIGS.get(ablation_name, {}).get('description', ''),
            'training_history': {
                'final_asr': history['asr'][-1] if history['asr'] else 0,
                'best_asr': max(history['asr']) if history['asr'] else 0,
                'mean_reward': float(np.mean(history['episode_rewards'][-1000:]))
            },
            'evaluation': eval_results
        }
    
    def run_all_ablations(
        self,
        train_episodes: int = 20000,
        eval_samples: int = 500,
        epsilon: float = 0.03,
        ablations: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run all ablation studies."""
        ablations = ablations or list(self.ABLATION_CONFIGS.keys())
        
        for name in ablations:
            try:
                self.results[name] = self.run_single_ablation(
                    name, train_episodes, eval_samples, epsilon
                )
            except Exception as e:
                logger.error(f"Ablation {name} failed: {e}")
                self.results[name] = {'error': str(e)}
        
        self._save_results()
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save ablation results."""
        path = self.output_dir / 'ablation_results.json'
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
        logger.info(f"Saved results to {path}")
    
    def _print_summary(self):
        """Print ablation summary table."""
        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)
        print(f"{'Ablation':<20} {'ASR':>10} {'SSIM':>10} {'Description':<30}")
        print("-"*70)
        
        for name, res in self.results.items():
            if 'error' in res:
                print(f"{name:<20} {'ERROR':>10} {'-':>10} {res['error'][:30]}")
            else:
                asr = res['evaluation']['metrics']['ASR']
                ssim = res['evaluation']['metrics']['SSIM']
                desc = res['description'][:30]
                print(f"{name:<20} {asr:>10.2%} {ssim:>10.4f} {desc}")
        print("="*70)


def run_ablation_study(
    victim_model_path: str,
    dataset_name: str = 'pathmnist',
    output_dir: str = 'results/ablation',
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for ablation studies."""
    runner = AblationRunner(victim_model_path, dataset_name, output_dir)
    return runner.run_all_ablations(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='Run MedSecure ablation studies')
    parser.add_argument('--victim-model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='pathmnist')
    parser.add_argument('--output-dir', type=str, default='results/ablation')
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--eval-samples', type=int, default=500)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--ablations', type=str, nargs='+', default=None)
    
    args = parser.parse_args()
    
    run_ablation_study(
        victim_model_path=args.victim_model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        train_episodes=args.episodes,
        eval_samples=args.eval_samples,
        epsilon=args.epsilon,
        ablations=args.ablations
    )


if __name__ == '__main__':
    main()
