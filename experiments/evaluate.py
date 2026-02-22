"""
Comprehensive evaluation of MedSecure agent.

Evaluates the trained RL agent against baseline attacks across
multiple metrics: ASR, image quality, efficiency, and diversity.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataset, get_dataloader
from models import get_victim_model
from rl import AdversarialAttackEnv, MedSecureAgent
from rl.hierarchical_env import HierarchicalAttackEnv, AttackStrategy
from rl.hierarchical_agent_sac import HierarchicalAgentSAC, create_hierarchical_agent_sac
from attacks import ATTACK_REGISTRY
from attacks.baselines import FGSM, PGD, CarliniWagner, AutoAttackWrapper
from attacks.frequency_attack import FrequencyAttack
from evaluation import (
    ComprehensiveMetrics,
    ComprehensiveDiversityMetrics,
    ComprehensiveEfficiencyMetrics,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for MedSecure.
    """
    
    def __init__(
        self,
        victim_model_path: str,
        agent_path: Optional[str] = None,
        dataset_name: str = 'pathmnist',
        device: Optional[torch.device] = None,
        output_dir: str = 'results/evaluation',
        algorithm: Optional[str] = None,
        strategies: Optional[List[str]] = None,
        force_strategy: Optional[str] = None
    ):
        """
        Args:
            victim_model_path: Path to victim model checkpoint
            agent_path: Path to trained agent checkpoint
            dataset_name: Dataset name
            device: Device for computation
            output_dir: Directory for saving results
            algorithm: RL algorithm ('sac' or 'ppo'), auto-detected if None
            strategies: Attack strategies (default: ['pixel', 'frequency'])
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm
        self.strategies = strategies or ['pixel', 'frequency']
        self.force_strategy = force_strategy
        
        STRATEGY_NAME_TO_IDX = {"pixel": 0, "frequency": 1, "square": 2}
        self.force_strategy_idx = STRATEGY_NAME_TO_IDX.get(force_strategy) if force_strategy else None
        
        # Load victim checkpoint to get metadata
        victim_checkpoint = torch.load(victim_model_path, map_location=self.device)
        
        # Get native flag BEFORE creating model
        self.native = victim_checkpoint.get('native', False)
        self.n_classes = victim_checkpoint['n_classes']
        
        # Load victim model with checkpoint (auto-detects native/small_input)
        self.victim_model = get_victim_model(
            victim_checkpoint['model_name'],
            num_classes=self.n_classes,
            checkpoint=victim_model_path,  # This auto-detects native and loads weights
            device=str(self.device)
        )
        self.victim_model.eval()
        
        # Get normalization stats from checkpoint
        self.normalize_stats = victim_checkpoint.get('normalize_stats', None)
        
        if self.native:
            logger.info(f"Native mode: 28x28 images")
            # Default normalization for native if not specified
            if self.normalize_stats is None:
                self.normalize_stats = {'mean': [0.5], 'std': [0.5]}
        else:
            logger.info(f"Upscale mode: 224x224 images")
            # Default to ImageNet normalization for upscaled if not specified
            if self.normalize_stats is None:
                self.normalize_stats = {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
        
        logger.info(f"Normalization stats: mean={self.normalize_stats['mean']}, std={self.normalize_stats['std']}")
        
        # Load test dataset with correct transforms
        self.test_dataset = get_dataset(
            dataset_name, 
            split='test', 
            download=True,
            native=self.native
        )
        self.test_loader = get_dataloader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        # Load RL agent if provided
        self.agent = None
        self.env = None
        if agent_path:
            self._load_agent(agent_path, self.algorithm)
        
        # Initialize metrics with normalization stats
        self.metrics = ComprehensiveMetrics(
            device=self.device,
            normalize_stats=self.normalize_stats
        )
        self.diversity = ComprehensiveDiversityMetrics(n_classes=self.n_classes)
        self.efficiency = ComprehensiveEfficiencyMetrics(device=self.device)
        
        logger.info(f"Initialized Evaluator")
        logger.info(f"  Victim model: {victim_checkpoint['model_name']}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Test samples: {len(self.test_dataset)}")
    
    def _load_agent(self, agent_path: str, algorithm: Optional[str] = None):
        """Load trained hierarchical SAC agent.
        
        Args:
            agent_path: Path to agent checkpoint directory
            algorithm: Ignored (kept for compatibility)
        """
        agent_path = Path(agent_path)
        
        # Agent path should be a directory containing sac_*.zip files, meta_controller_*, stats.json
        if not agent_path.is_dir():
            # Try parent directory
            if agent_path.parent.is_dir():
                agent_path = agent_path.parent
            else:
                raise FileNotFoundError(f"Agent directory not found: {agent_path}")
        
        # Check for SAC files - support both old format (sac_controller.zip) and new (sac_pixel.zip, sac_frequency.zip)
        has_old_format = (agent_path / "sac_controller.zip").exists()
        has_new_format = (agent_path / "sac_pixel.zip").exists() or (agent_path / "sac_frequency.zip").exists()
        
        if not has_old_format and not has_new_format:
            raise FileNotFoundError(f"No SAC checkpoint files found in {agent_path}. Expected sac_pixel.zip/sac_frequency.zip or sac_controller.zip")
        
        if has_old_format and not has_new_format:
            raise FileNotFoundError(f"Old checkpoint format (sac_controller.zip) found. Please retrain with the new SAC-separati architecture.")
        
        logger.info(f"Loading hierarchical SAC agent from {agent_path}")
        
        # Create dataloader for environment
        train_dataset = get_dataset(
            self.dataset_name,
            split="train",
            native=self.native,
            download=True
        )
        train_loader = get_dataloader(
            train_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0
        )
        
        # Environment config
        env_config = {
            "max_steps_per_image": 5,
            "rl": {
                "action_space": {
                    "epsilon_range": [0.0002, 0.1],
                    "iterations_range": [1, 100],
                    "step_size_range": [0.001, 0.1]
                }
            }
        }
        
        # Initialize hierarchical environment in "separated" mode
        self.env = HierarchicalAttackEnv(
            victim_model=self.victim_model,
            attack_registry=ATTACK_REGISTRY,
            dataloader=train_loader,
            config=env_config,
            device=str(self.device),
            mode="separated",  # Important: SAC agent uses separated mode
            allowed_strategies=self.strategies,
            normalize_stats=self.normalize_stats
        )
        
        # Load state extractor statistics if available
        state_stats_path = agent_path / "state_extractor_stats.pt"
        if self.env.state_extractor.load_stats(str(state_stats_path)):
            logger.info(f"  Loaded state extractor stats from {state_stats_path}")
            self.env.state_extractor.eval()  # Don't update stats during evaluation
        else:
            logger.warning(f"  State extractor stats not found at {state_stats_path}")
            logger.warning(f"  Running warmup to estimate statistics...")
            self._warmup_state_extractor(n_samples=500)
        
        # Agent config
        agent_config = {
            "sac_lr": 3e-4,
            "sac_buffer_size": 100000,
            "sac_batch_size": 256,
            "sac_learning_starts": 1000,
            "meta_lr": 1e-4,
            "meta_buffer_size": 10000,
            "meta_batch_size": 32,
            "meta_epsilon_start": 0.0,  # No exploration for eval
            "meta_epsilon_min": 0.0,
            "meta_epsilon_decay": 1.0,
        }
        
        # Create agent
        self.agent = create_hierarchical_agent_sac(
            self.env, 
            agent_config, 
            str(self.device)
        )
        
        # Load checkpoint
        self.agent.load(str(agent_path))
        
        # Set to evaluation mode (no exploration)
        self.agent.meta_controller.epsilon = 0.0
        
        logger.info(f"Hierarchical SAC agent loaded successfully")
        logger.info(f"  Timesteps trained: {self.agent.total_timesteps}")
        logger.info(f"  Episodes: {self.agent.episode_count}")
        logger.info(f"  Strategy usage: {self.agent.strategy_usage}")
    
    def _warmup_state_extractor(self, n_samples: int = 500):
        """Warmup state extractor to estimate normalization statistics.
        
        This is needed when loading an agent without saved statistics.
        Collects raw states and computes statistics at the end.
        """
        logger.info(f"  Warming up state extractor with {n_samples} samples...")
        
        # Temporarily disable normalization to collect raw states
        original_mean = self.env.state_extractor._mean.clone()
        original_var = self.env.state_extractor._var.clone()
        
        # Set to "pass-through" mode: mean=0, var=1, no updates
        self.env.state_extractor._mean = torch.zeros(
            self.env.state_extractor.STATE_DIM, 
            device=self.env.state_extractor.device
        )
        self.env.state_extractor._var = torch.ones(
            self.env.state_extractor.STATE_DIM, 
            device=self.env.state_extractor.device
        )
        self.env.state_extractor.eval()  # Don't update stats
        
        # Collect raw states
        raw_states = []
        count = 0
        for images, labels in self.test_loader:
            if count >= n_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                if count >= n_samples:
                    break
                
                # Reset env with this image - this computes state
                obs, _ = self.env.reset(options={'image': images[i:i+1], 'label': labels[i:i+1]})
                raw_states.append(torch.tensor(obs[:21], device=self.env.state_extractor.device))
                count += 1
        
        # Stack all states and compute statistics
        all_states = torch.stack(raw_states)  # [n_samples, 21]
        computed_mean = all_states.mean(dim=0)
        computed_var = all_states.var(dim=0, unbiased=False)
        
        # Set the computed statistics
        self.env.state_extractor._mean = computed_mean
        self.env.state_extractor._var = torch.maximum(computed_var, torch.tensor(1e-4, device=self.env.state_extractor.device))
        self.env.state_extractor._count = count
        
        logger.info(f"  State extractor warmed up with {count} samples")
        logger.info(f"  Mean range: [{computed_mean.min():.4f}, {computed_mean.max():.4f}]")
        logger.info(f"  Var range: [{self.env.state_extractor._var.min():.4f}, {self.env.state_extractor._var.max():.4f}]")
        
        # Save stats for future use
        stats_save_path = self.output_dir / "state_extractor_stats.pt"
        self.env.state_extractor.save_stats(str(stats_save_path))
        logger.info(f"  Saved stats to: {stats_save_path}")
    
    def evaluate_agent(
        self,
        n_samples: int = 1000,
        epsilon: float = 0.03
    ) -> Dict[str, Any]:
        """
        Evaluate the hierarchical RL agent.
        
        Args:
            n_samples: Number of samples to evaluate
            epsilon: Perturbation budget
        
        Returns:
            Evaluation results
        """
        if self.agent is None:
            raise ValueError("No agent loaded. Provide agent_path in constructor.")
        
        logger.info(f"Evaluating hierarchical RL agent on {n_samples} samples...")
        logger.info(f"  Epsilon budget: {epsilon}")
        if self.force_strategy_idx is not None:
            logger.info(f"  ⚠ FORCED STRATEGY: {self.force_strategy.upper()} (ablation mode)")
        
        # Reset metrics
        self.metrics.reset()
        self.diversity.reset()
        self.efficiency.reset()
        
        # Update environment epsilon - MUST match training range
        # Training uses: [0.001, target_epsilon]
        self.env.set_curriculum_config(
            epsilon_range=[0.001, epsilon],
            curriculum_phase=3  # Expert phase
        )
        
        logger.info(f"  Epsilon budget: {epsilon}")
        logger.info(f"  Environment epsilon_range after set: {self.env.epsilon_range}")
        
        successes = 0
        total = 0
        strategy_counts = {s.upper(): 0 for s in self.strategies}
        strategy_successes = {s.upper(): 0 for s in self.strategies}
        
        # Per-sample tracking for paper figures
        epsilon_per_sample = []
        confidence_per_sample = []
        strategy_per_sample = []
        
        # Open file for epsilon logging
        epsilon_log_path = self.output_dir / "epsilon_per_sample.csv"
        epsilon_log = open(epsilon_log_path, 'w')
        epsilon_log.write("sample_idx,strategy,epsilon_raw,epsilon_decoded,epsilon_actual,success\n")
        
        logger.info(f"Allowed strategies: {[s.name for s in self.env.allowed_strategies]}")
        logger.info(f"Epsilon log: {epsilon_log_path}")
        
        # Use fixed_samples if available (for fair comparison across methods)
        if hasattr(self, 'fixed_samples') and self.fixed_samples:
            sample_list = self.fixed_samples[:n_samples]
            logger.info(f"  Using {len(sample_list)} pre-selected fixed samples")
            use_fixed = True
        else:
            sample_list = range(n_samples)
            use_fixed = False
        
        for sample_idx, sample_data in enumerate(tqdm(sample_list, desc="Evaluating agent")):
            if use_fixed:
                images, labels = sample_data
                obs, info = self.env.reset(options={'image': images, 'label': labels})
            else:
                obs, info = self.env.reset()
            
            # Get original image and prediction
            original = self.env.current_image.clone()
            original_label = self.env.current_label.clone()
            
            with torch.no_grad():
                original_output = self.victim_model(original)
                original_pred = original_output.argmax(dim=1)
                # Capture original confidence for paper figures
                original_conf = torch.nn.functional.softmax(original_output, dim=1).max().item()
            
            # Skip if model already wrong (only needed when not using fixed samples)
            if not use_fixed and original_pred.item() != original_label.item():
                continue
            
            self.efficiency.start_attack()
            
            # Run episode with hierarchical agent
            done = False
            base_state = obs[:21]  # First 21 dims are base state (includes epsilon_budget)
            
            # Meta-controller selects strategy (deterministic for eval)
            if self.force_strategy_idx is not None:
                current_strategy = self.force_strategy_idx
            else:
                current_strategy = self.agent.meta_controller.select_strategy(base_state, deterministic=True)
            self.env.set_strategy(current_strategy)
            
            last_strategy = AttackStrategy(current_strategy).name
            episode_queries = 0
            episode_steps = 0
            
            
            debug_params = []
            
            while not done:
                params, _ = self.agent.controllers[current_strategy].predict(base_state, deterministic=True)
                
                # Track params (only first step needed for epsilon)
                if episode_steps == 0:
                    debug_params.append({
                        'step': episode_steps,
                        'epsilon_raw': params[0],
                        'params_raw': params[:4].tolist()
                    })
                
                # Execute action - in separated mode, just pass params
                obs, reward, terminated, truncated, info = self.env.step(params)
                done = terminated or truncated
                
                base_state = obs[:21]  # STATE_DIM = 21
                last_strategy = info.get('strategy', last_strategy)
                episode_queries += info.get('queries', 1)
                episode_steps += 1
            
            # Get adversarial result FIRST (before debug)
            adversarial = self.env.current_adversarial
            if adversarial is None:
                adversarial = original
            
            # Get epsilon chosen by model (decoded)
            eps_used = info.get('params', {}).get('epsilon', 0)
            
            # Compute actual perturbation in [0,1] space
            if self.normalize_stats:
                mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(self.device)
                std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(self.device)
                orig_denorm = original * std + mean
                adv_denorm = adversarial * std + mean
                actual_pert = (adv_denorm - orig_denorm).abs().max().item()
            else:
                actual_pert = (adversarial - original).abs().max().item()
            
            # Print epsilon for EVERY sample
            raw_eps = debug_params[0]['epsilon_raw'] if debug_params else 0
            # Write to CSV file
            epsilon_log.write(f"{sample_idx},{last_strategy},{raw_eps:.6f},{eps_used:.6f},{actual_pert:.6f},{info.get('success', False)}\n")
            epsilon_log.flush()
            
            with torch.no_grad():
                adv_output = self.victim_model(adversarial)
                adv_pred = adv_output.argmax(dim=1)
            
            # Check success
            success = (original_pred.item() == original_label.item() and 
                      adv_pred.item() != original_label.item())
            
            # Compute perturbation
            perturbation = adversarial - original
            pert_norm = perturbation.abs().max().item()
            
            self.efficiency.end_attack(success, pert_norm)
            self.efficiency.queries.count_forward(episode_queries)
            
            # Update metrics
            self.metrics.update(
                original=original,
                adversarial=adversarial,
                original_preds=original_pred,
                adversarial_preds=adv_pred,
                true_labels=original_label
            )
            
            if success:
                successes += 1
            total += 1
            
            # Track strategy usage and success
            if last_strategy in strategy_counts:
                strategy_counts[last_strategy] += 1
                if success:
                    strategy_successes[last_strategy] += 1
            
            # Track per-sample data for paper figures
            # Use epsilon_decoded (the epsilon chosen by the agent)
            epsilon_per_sample.append(float(eps_used))
            confidence_per_sample.append(float(original_conf))
            strategy_per_sample.append(last_strategy)
        
        # Close epsilon log file
        epsilon_log.close()
        logger.info(f"Epsilon log saved to: {epsilon_log_path}")
        
        # Compute final metrics
        metrics_summary = self.metrics.summary()
        efficiency_summary = self.efficiency.summary()
        
        # Fix queries per attack
        if total > 0:
            efficiency_summary['mean_queries_per_attack'] = efficiency_summary['total_queries'] / total
        
        # Compute per-strategy ASR
        strategy_asr = {}
        for strategy in strategy_counts:
            if strategy_counts[strategy] > 0:
                strategy_asr[strategy] = strategy_successes[strategy] / strategy_counts[strategy]
            else:
                strategy_asr[strategy] = 0.0
        
        results = {
            'method': 'MedSecure (Hierarchical RL)',
            'n_samples': total,
            'epsilon': epsilon,
            'metrics': metrics_summary,
            'efficiency': efficiency_summary,
            'strategy_distribution': strategy_counts,
            'strategy_asr': strategy_asr,
            'strategy_successes': strategy_successes,
            # Per-sample data for paper figures
            'epsilon_per_sample': epsilon_per_sample,
            'confidence_per_sample': confidence_per_sample,
            'strategy_per_sample': strategy_per_sample,
        }
        
        logger.info(f"Evaluation complete:")
        logger.info(f"  ASR: {metrics_summary['ASR']:.2%}")
        logger.info(f"  SSIM: {metrics_summary['SSIM']:.4f}")
        logger.info(f"  PSNR: {metrics_summary['PSNR']:.2f} dB")
        logger.info(f"  Strategy distribution: {strategy_counts}")
        logger.info(f"  Strategy ASR: { {k: f'{v:.1%}' for k, v in strategy_asr.items()} }")
        
        return results
    
    def evaluate_agent_on_samples(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        epsilon: float = 0.03
    ) -> Dict[str, Any]:
        """
        Evaluate the hierarchical RL agent on specific samples.
        
        Args:
            samples: List of (image, label) tuples
            epsilon: Perturbation budget
        
        Returns:
            Evaluation results
        """
        if self.agent is None:
            raise ValueError("No agent loaded. Provide agent_path in constructor.")
        
        n_samples = len(samples)
        logger.info(f"Evaluating hierarchical RL agent on {n_samples} pre-selected samples...")
        
        # Reset metrics
        self.metrics.reset()
        self.diversity.reset()
        self.efficiency.reset()
        
        # Update environment epsilon - MUST match training range
        self.env.set_curriculum_config(
            epsilon_range=[0.001, epsilon],
            curriculum_phase=3
        )
        
        successes = 0
        total = 0
        strategy_counts = {s.upper(): 0 for s in self.strategies}
        strategy_successes = {s.upper(): 0 for s in self.strategies}
        
        # Budget verification tracking
        pert_pixel_list = []
        pert_norm_list = []
        budget_violations = 0
        
        # Per-sample tracking for paper figures
        epsilon_per_sample = []
        confidence_per_sample = []
        strategy_per_sample = []
        
        # Open file for epsilon logging
        epsilon_log_path = self.output_dir / "epsilon_per_sample.csv"
        epsilon_log = open(epsilon_log_path, 'w')
        epsilon_log.write("sample_idx,strategy,epsilon_raw,epsilon_decoded,epsilon_actual,success\n")
        logger.info(f"Epsilon log: {epsilon_log_path}")
        
        for sample_idx, (image, label) in enumerate(tqdm(samples, desc="Evaluating MedSecure")):
            # Reset env with specific image
            obs, info = self.env.reset(options={'image': image, 'label': label})
            
            original = self.env.current_image.clone()
            original_label = self.env.current_label.clone()
            
            with torch.no_grad():
                original_output = self.victim_model(original)
                original_pred = original_output.argmax(dim=1)
                # Capture original confidence for paper figures
                original_conf = torch.nn.functional.softmax(original_output, dim=1).max().item()
            
            self.efficiency.start_attack()
            
            # Run episode with hierarchical agent
            done = False
            base_state = obs[:21]
            
            if sample_idx < 10:
                logger.info(f"\n=== Sample {sample_idx} DEBUG ===")
                logger.info(f"  image.shape={image.shape}, image.min={image.min():.4f}, image.max={image.max():.4f}")
                logger.info(f"  obs.shape={obs.shape}, base_state.shape={base_state.shape}")
                logger.info(f"  base_state[:5]={base_state[:5]}")
                logger.info(f"  base_state min={base_state.min():.4f}, max={base_state.max():.4f}")
            
            # Meta-controller selects strategy
            if self.force_strategy_idx is not None:
                current_strategy = self.force_strategy_idx
            else:
                current_strategy = self.agent.meta_controller.select_strategy(base_state, deterministic=True)
            self.env.set_strategy(current_strategy)
            
            last_strategy = AttackStrategy(current_strategy).name
            episode_queries = 0
            episode_steps = 0
            debug_params = []
            
            while not done:
                params, _ = self.agent.controllers[current_strategy].predict(base_state, deterministic=True)
                
                if episode_steps == 0:
                    debug_params.append({
                        'epsilon_raw': params[0],
                        'params_raw': params[:4].tolist()
                    })
                
                obs, reward, terminated, truncated, info = self.env.step(params)
                done = terminated or truncated
                
                base_state = obs[:21]
                last_strategy = info.get('strategy', last_strategy)
                episode_queries += info.get('queries', 1)
                episode_steps += 1
            
            adversarial = self.env.current_adversarial
            if adversarial is None:
                adversarial = original
            
            # Get epsilon chosen by model (decoded)
            eps_used = info.get('params', {}).get('epsilon', 0)
            
            # === BUDGET VERIFICATION ===
            pert_normalized = (adversarial - original).abs().max().item()
            pert_norm_list.append(pert_normalized)
            
            if self.normalize_stats:
                mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(self.device)
                std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(self.device)
                orig_pixel = original * std + mean
                adv_pixel = adversarial * std + mean
                pert_pixel = (adv_pixel - orig_pixel).abs().max().item()
            else:
                pert_pixel = pert_normalized
            pert_pixel_list.append(pert_pixel)
            
            # Write to CSV file
            raw_eps = debug_params[0]['epsilon_raw'] if debug_params else 0
            epsilon_log.write(f"{sample_idx},{last_strategy},{raw_eps:.6f},{eps_used:.6f},{pert_pixel:.6f},{info.get('success', False)}\n")
            epsilon_log.flush()
            
            if pert_pixel > epsilon * 1.01:
                budget_violations += 1
            
            with torch.no_grad():
                adv_output = self.victim_model(adversarial)
                adv_pred = adv_output.argmax(dim=1)
            
            success = (original_pred.item() == original_label.item() and 
                      adv_pred.item() != original_label.item())
            
            perturbation = adversarial - original
            pert_norm = perturbation.abs().max().item()
            
            self.efficiency.end_attack(success, pert_norm)
            self.efficiency.queries.count_forward(episode_queries)
            
            self.metrics.update(
                original=original,
                adversarial=adversarial,
                original_preds=original_pred,
                adversarial_preds=adv_pred,
                true_labels=original_label
            )
            
            if success:
                successes += 1
            total += 1
            
            if last_strategy in strategy_counts:
                strategy_counts[last_strategy] += 1
                if success:
                    strategy_successes[last_strategy] += 1
            
            # Track per-sample data for paper figures
            # Use epsilon_decoded (the epsilon chosen by the agent)
            epsilon_per_sample.append(float(eps_used))
            confidence_per_sample.append(float(original_conf))
            strategy_per_sample.append(last_strategy)
        
        # === LOG BUDGET VERIFICATION ===
        import statistics
        if pert_pixel_list:
            logger.info(f"\n  === MedSecure BUDGET VERIFICATION ===")
            logger.info(f"  Epsilon budget (pixel space): {epsilon}")
            logger.info(f"  L-inf pixel space: mean={statistics.mean(pert_pixel_list):.6f}, "
                        f"max={max(pert_pixel_list):.6f}, min={min(pert_pixel_list):.6f}")
            logger.info(f"  L-inf normalized:  mean={statistics.mean(pert_norm_list):.6f}, "
                        f"max={max(pert_norm_list):.6f}")
            logger.info(f"  Budget violations: {budget_violations}/{total} "
                        f"({100*budget_violations/max(1,total):.1f}%)")
            logger.info(f"  ========================================\n")
        
        # Close epsilon log file
        epsilon_log.close()
        logger.info(f"Epsilon log saved to: {epsilon_log_path}")
        
        # Compute final metrics
        metrics_summary = self.metrics.summary()
        efficiency_summary = self.efficiency.summary()
        
        if total > 0:
            efficiency_summary['mean_queries_per_attack'] = efficiency_summary['total_queries'] / total
        
        strategy_asr = {}
        for strategy in strategy_counts:
            if strategy_counts[strategy] > 0:
                strategy_asr[strategy] = strategy_successes[strategy] / strategy_counts[strategy]
            else:
                strategy_asr[strategy] = 0.0
        
        results = {
            'method': 'MedSecure (Hierarchical RL)',
            'n_samples': total,
            'epsilon': epsilon,
            'metrics': metrics_summary,
            'efficiency': efficiency_summary,
            'strategy_distribution': strategy_counts,
            'strategy_asr': strategy_asr,
            'strategy_successes': strategy_successes,
            # Per-sample data for paper figures
            'epsilon_per_sample': epsilon_per_sample,
            'confidence_per_sample': confidence_per_sample,
            'strategy_per_sample': strategy_per_sample,
        }
        
        logger.info(f"Evaluation complete: ASR={metrics_summary['ASR']:.2%}")
        
        return results
    
    def evaluate_baseline(
        self,
        attack_name: str,
        n_samples: int = 1000,
        epsilon: float = 0.03,
        **attack_kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a baseline attack.
        
        Args:
            attack_name: Attack name ('fgsm', 'pgd', 'freq', 'cw')
            n_samples: Number of samples
            epsilon: Perturbation budget
            **attack_kwargs: Additional attack parameters
        
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {attack_name.upper()} baseline...")
        logger.info(f"  Using epsilon: {epsilon}")
        
        # Initialize attack with normalization stats
        attack_name_lower = attack_name.lower()
        if attack_name_lower == 'fgsm':
            attack = FGSM(
                self.victim_model, 
                epsilon=epsilon, 
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
            logger.info(f"  FGSM initialized with epsilon={attack.epsilon}")
        elif attack_name_lower == 'pgd':
            attack = PGD(
                self.victim_model, 
                epsilon=epsilon,
                iterations=attack_kwargs.get('iterations', 40),
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
            logger.info(f"  PGD initialized with epsilon={attack.epsilon}")
        elif attack_name_lower == 'freq':
            attack = FrequencyAttack(
                self.victim_model, 
                epsilon=epsilon,
                iterations=attack_kwargs.get('iterations', 40),
                target_bands='low',
                band_ratio=0.25,
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
            logger.info(f"  FrequencyAttack initialized with epsilon={attack.epsilon}")
        elif attack_name_lower in ['cw', 'c&w']:
            attack = CarliniWagner(
                self.victim_model,
                confidence=attack_kwargs.get('confidence', 0),
                max_iterations=attack_kwargs.get('max_iterations', 1000),
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
        elif attack_name_lower in ['autoattack', 'aa']:
            attack = AutoAttackWrapper(
                self.victim_model,
                epsilon=epsilon,
                norm='Linf',
                version='standard',
                device=str(self.device),
                normalize_stats=self.normalize_stats,
                verbose=False
            )
            logger.info(f"  AutoAttack initialized with epsilon={epsilon}")
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
        
        # Reset metrics
        self.metrics.reset()
        self.efficiency.reset()
        
        samples_processed = 0
        total_success = 0
        total_ssim = 0.0
        total_psnr = 0.0
        
        # Use fixed_samples if available (for fair comparison across methods)
        if hasattr(self, 'fixed_samples') and self.fixed_samples:
            sample_iterator = [(img, lbl) for img, lbl in self.fixed_samples[:n_samples]]
            logger.info(f"  Using {len(sample_iterator)} pre-selected fixed samples")
        else:
            # Fallback: iterate from test_loader
            sample_iterator = []
            for images, labels in self.test_loader:
                for i in range(images.size(0)):
                    sample_iterator.append((images[i:i+1], labels[i:i+1]))
                    if len(sample_iterator) >= n_samples:
                        break
                if len(sample_iterator) >= n_samples:
                    break
        
        for images, labels in tqdm(sample_iterator, desc=f"Evaluating {attack_name}"):
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            batch_size = images.size(0)
            
            # Get original predictions
            with torch.no_grad():
                original_outputs = self.victim_model(images)
                original_preds = original_outputs.argmax(dim=1)
            
            # Run attack
            self.efficiency.start_attack()
            result = attack.attack(images, labels)
            adversarial = result.adversarial
            
            if samples_processed < 3:
                # Denormalize to get true perturbation
                if self.normalize_stats:
                    mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(self.device)
                    std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(self.device)
                    orig_denorm = images * std + mean
                    adv_denorm = adversarial * std + mean
                    actual_pert = (adv_denorm - orig_denorm).abs().max().item()
                else:
                    actual_pert = (adversarial - images).abs().max().item()
                logger.info(f"  Sample {samples_processed}: L-inf perturbation in [0,1] = {actual_pert:.4f} (expected <= {epsilon})")
            
            # Count queries (already tracked by attack internally)
            self.efficiency.queries.count_forward(result.queries)
            
            # Get adversarial predictions
            with torch.no_grad():
                adversarial_outputs = self.victim_model(adversarial)
                adversarial_preds = adversarial_outputs.argmax(dim=1)
            
            # Compute success per sample
            originally_correct = (original_preds == labels)
            now_incorrect = (adversarial_preds != labels)
            success = (originally_correct & now_incorrect)
            
            perturbation_norm = (adversarial - images).view(batch_size, -1).norm(p=float('inf'), dim=1).mean().item()
            self.efficiency.end_attack(success.any().item(), perturbation_norm)
            
            # Update metrics
            self.metrics.update(
                original=images,
                adversarial=adversarial,
                original_preds=original_preds,
                adversarial_preds=adversarial_preds,
                true_labels=labels
            )
            
            total_success += success.sum().item()
            samples_processed += batch_size
        
        # Get summary
        metrics_summary = self.metrics.summary()
        efficiency_summary = self.efficiency.summary()
        
        # Fix mean_queries_per_attack: divide by samples, not batches
        if samples_processed > 0:
            efficiency_summary['mean_queries_per_attack'] = efficiency_summary['total_queries'] / samples_processed
        
        results = {
            'method': attack_name.upper(),
            'n_samples': samples_processed,
            'epsilon': epsilon,
            'metrics': metrics_summary,
            'efficiency': efficiency_summary
        }
        
        return results
    
    def evaluate_baseline_on_samples(
        self,
        attack_name: str,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        epsilon: float = 0.03,
        **attack_kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a baseline attack on specific samples.
        
        Args:
            attack_name: Attack name ('fgsm', 'pgd', 'freq', 'cw')
            samples: List of (image, label) tuples
            epsilon: Perturbation budget
            **attack_kwargs: Additional attack parameters
        
        Returns:
            Evaluation results
        """
        n_samples = len(samples)
        logger.info(f"Evaluating {attack_name.upper()} on {n_samples} pre-selected samples...")
        
        # Initialize attack with normalization stats
        attack_name_lower = attack_name.lower()
        if attack_name_lower == 'fgsm':
            attack = FGSM(
                self.victim_model, 
                epsilon=epsilon, 
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
        elif attack_name_lower == 'pgd':
            attack = PGD(
                self.victim_model, 
                epsilon=epsilon,
                iterations=attack_kwargs.get('iterations', 40),
                step_size=attack_kwargs.get('step_size', epsilon/10),
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
        elif attack_name_lower == 'freq':
            attack = FrequencyAttack(
                self.victim_model, 
                epsilon=epsilon,
                iterations=attack_kwargs.get('iterations', 40),
                target_bands='low',
                band_ratio=0.25,
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
        elif attack_name_lower == 'cw':
            attack = CarliniWagner(
                self.victim_model,
                confidence=attack_kwargs.get('confidence', 0),
                learning_rate=attack_kwargs.get('learning_rate', 0.01),
                max_iterations=attack_kwargs.get('max_iterations', 100),
                device=str(self.device),
                normalize_stats=self.normalize_stats
            )
        elif attack_name_lower in ['autoattack', 'aa']:
            attack = AutoAttackWrapper(
                self.victim_model,
                epsilon=epsilon,
                norm='Linf',
                version='standard',
                device=str(self.device),
                normalize_stats=self.normalize_stats,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
        
        # Reset metrics
        self.metrics.reset()
        self.efficiency.reset()
        
        total_success = 0
        samples_processed = 0
        
        # Budget verification tracking
        pert_pixel_list = []   # L-inf in [0,1] pixel space
        pert_norm_list = []    # L-inf in normalized space
        budget_violations = 0
        
        for image, label in tqdm(samples, desc=f"Evaluating {attack_name.upper()}"):
            image = image.to(self.device)
            label = label.to(self.device).squeeze()
            
            if label.dim() == 0:
                label = label.unsqueeze(0)
            
            # Get original predictions
            with torch.no_grad():
                original_output = self.victim_model(image)
                original_pred = original_output.argmax(dim=1)
            
            # Run attack
            self.efficiency.start_attack()
            result = attack.attack(image, label)
            adversarial = result.adversarial
            
            # Count queries
            self.efficiency.queries.count_forward(result.queries)
            
            # === BUDGET VERIFICATION ===
            # L-inf in normalized space
            pert_normalized = (adversarial - image).abs().max().item()
            pert_norm_list.append(pert_normalized)
            
            # L-inf in pixel space [0,1] (denormalize first)
            if self.normalize_stats:
                mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(self.device)
                std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(self.device)
                orig_pixel = image * std + mean
                adv_pixel = adversarial * std + mean
                pert_pixel = (adv_pixel - orig_pixel).abs().max().item()
            else:
                pert_pixel = pert_normalized
            pert_pixel_list.append(pert_pixel)
            
            if pert_pixel > epsilon * 1.01:  # 1% tolerance
                budget_violations += 1
            
            # Get adversarial predictions
            with torch.no_grad():
                adversarial_output = self.victim_model(adversarial)
                adversarial_pred = adversarial_output.argmax(dim=1)
            
            # Compute success
            success = (original_pred.item() == label.item() and 
                      adversarial_pred.item() != label.item())
            
            perturbation_norm = (adversarial - image).abs().max().item()
            self.efficiency.end_attack(success, perturbation_norm)
            
            # Update metrics
            self.metrics.update(
                original=image,
                adversarial=adversarial,
                original_preds=original_pred,
                adversarial_preds=adversarial_pred,
                true_labels=label
            )
            
            if success:
                total_success += 1
            samples_processed += 1
        
        # === LOG BUDGET VERIFICATION ===
        import statistics
        if pert_pixel_list:
            logger.info(f"\n  === {attack_name.upper()} BUDGET VERIFICATION ===")
            logger.info(f"  Epsilon budget (pixel space): {epsilon}")
            logger.info(f"  L-inf pixel space: mean={statistics.mean(pert_pixel_list):.6f}, "
                        f"max={max(pert_pixel_list):.6f}, min={min(pert_pixel_list):.6f}")
            logger.info(f"  L-inf normalized:  mean={statistics.mean(pert_norm_list):.6f}, "
                        f"max={max(pert_norm_list):.6f}")
            logger.info(f"  Budget violations: {budget_violations}/{samples_processed} "
                        f"({100*budget_violations/max(1,samples_processed):.1f}%)")
            logger.info(f"  ========================================\n")
        
        # Get summary
        metrics_summary = self.metrics.summary()
        efficiency_summary = self.efficiency.summary()
        
        if samples_processed > 0:
            efficiency_summary['mean_queries_per_attack'] = efficiency_summary['total_queries'] / samples_processed
        
        results = {
            'method': attack_name.upper(),
            'n_samples': samples_processed,
            'epsilon': epsilon,
            'metrics': metrics_summary,
            'efficiency': efficiency_summary
        }
        
        logger.info(f"  {attack_name.upper()} complete: ASR={metrics_summary['ASR']:.2%}")
        
        return results
    
    def _select_evaluation_samples(self, n_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Pre-select samples for evaluation (same for all methods).
        
        Only selects samples where model prediction is correct.
        
        Args:
            n_samples: Number of samples to select
            
        Returns:
            List of (image, label) tuples
        """
        logger.info(f"Pre-selecting {n_samples} evaluation samples...")
        
        samples = []
        for images, labels in self.test_loader:
            if len(samples) >= n_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Check each sample in batch
            with torch.no_grad():
                outputs = self.victim_model(images)
                preds = outputs.argmax(dim=1)
            
            for i in range(images.size(0)):
                if len(samples) >= n_samples:
                    break
                # Only keep correctly classified samples
                if preds[i].item() == labels[i].item():
                    samples.append((images[i:i+1].clone(), labels[i:i+1].clone()))
        
        logger.info(f"  Selected {len(samples)} samples (correctly classified)")
        return samples
    
    def compare_all(
        self,
        n_samples: int = 1000,
        epsilon: float = 0.03,
        include_cw: bool = False,
        include_autoattack: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare agent against all baselines on the SAME samples.
        
        Args:
            n_samples: Samples per method
            epsilon: Perturbation budget
            include_cw: Include C&W (slow)
            include_autoattack: Include AutoAttack (very slow)
        
        Returns:
            Dictionary mapping method to results
        """
        # Pre-select samples (same for all methods)
        eval_samples = self._select_evaluation_samples(n_samples)
        
        all_results = {}
        
        # Evaluate agent if available
        if self.agent is not None:
            all_results['MedSecure'] = self.evaluate_agent_on_samples(eval_samples, epsilon)
        
        # Evaluate baselines on same samples
        for attack_name in ['fgsm', 'pgd', 'freq']:
            all_results[attack_name.upper()] = self.evaluate_baseline_on_samples(
                attack_name, eval_samples, epsilon
            )
        
        if include_cw:
            all_results['C&W'] = self.evaluate_baseline_on_samples('cw', eval_samples, epsilon)
        
        if include_autoattack:
            all_results['AutoAttack'] = self.evaluate_baseline_on_samples('autoattack', eval_samples, epsilon)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"{'Method':<15} {'ASR':>10} {'SSIM':>10} {'PSNR':>10} {'Queries':>10}")
        print("-"*70)
        
        for method, res in results.items():
            asr = res['metrics'].get('ASR', 0)
            ssim = res['metrics'].get('SSIM', 0)
            psnr = res['metrics'].get('PSNR', 0)
            queries = res['efficiency'].get('mean_queries_per_attack', 0)
            
            print(f"{method:<15} {asr:>10.2%} {ssim:>10.4f} {psnr:>10.2f} {queries:>10.1f}")
        
        print("="*70)
        
        # Print strategy distribution and ASR for MedSecure
        if 'MedSecure' in results and 'strategy_distribution' in results['MedSecure']:
            print("\nMedSecure Strategy Performance:")
            print("-"*50)
            print(f"  {'Strategy':<12} {'Count':>8} {'Usage':>8} {'ASR':>10}")
            print("-"*50)
            
            dist = results['MedSecure']['strategy_distribution']
            asr = results['MedSecure'].get('strategy_asr', {})
            total = sum(dist.values())
            
            for strategy in sorted(dist.keys()):
                count = dist.get(strategy, 0)
                pct = count / total * 100 if total > 0 else 0
                strategy_asr = asr.get(strategy, 0)
                print(f"  {strategy:<12} {count:>8} {pct:>7.1f}% {strategy_asr:>9.1%}")
            
            print("-"*50)
            print()


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate MedSecure agent')
    
    parser.add_argument('--victim-model', type=str, required=True,
                       help='Path to victim model checkpoint')
    parser.add_argument('--agent', type=str, default=None,
                       help='Path to agent checkpoint')
    parser.add_argument('--dataset', type=str, default='pathmnist',
                       help='Dataset name')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Perturbation budget')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Only evaluate baselines')
    parser.add_argument('--include-cw', action='store_true',
                       help='Include C&W attack (slow)')
    parser.add_argument('--include-autoattack', action='store_true',
                       help='Include AutoAttack (very slow, pip install autoattack)')
    parser.add_argument('--algorithm', type=str, default=None, choices=['sac', 'ppo'],
                       help='RL algorithm (auto-detected if not specified)')
    parser.add_argument('--strategies', nargs='+', default=['pixel', 'frequency'],
                       choices=['pixel', 'frequency', 'square'],
                       help='Attack strategies (default: pixel frequency)')
    parser.add_argument('--force-strategy', type=str, default=None,
                       choices=['pixel', 'frequency', 'square'],
                       help='Force a specific strategy for all samples (ablation study)')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(
        victim_model_path=args.victim_model,
        agent_path=args.agent if not args.baselines_only else None,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        algorithm=args.algorithm,
        strategies=args.strategies,
        force_strategy=args.force_strategy
    )
    
    results = evaluator.compare_all(
        n_samples=args.n_samples,
        epsilon=args.epsilon,
        include_cw=args.include_cw,
        include_autoattack=args.include_autoattack
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    results_path = Path(args.output_dir) / 'evaluation_results.json'
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()