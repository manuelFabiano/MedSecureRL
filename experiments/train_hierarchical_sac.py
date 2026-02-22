#!/usr/bin/env python3
"""
Training script for Hierarchical RL Agent with SAC Controller.

This script trains a hierarchical agent that:
1. Uses a Meta-Controller (DQN) to select attack strategies
2. Uses a Controller (SAC) to optimize attack parameters

Key features:
- SAC for stable continuous action learning
- Automatic entropy tuning (no manual exploration issues)
- Curriculum learning for strategies
- Adaptive epsilon: agent learns to use minimum epsilon needed
- Comprehensive logging and checkpointing

Usage:
    python train_hierarchical_sac.py \
        --model-path checkpoints/resnet18_pathmnist_best.pth \
        --dataset pathmnist \
        --timesteps 100000 \
        --curriculum
"""

import argparse
import json
import sys
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict

# Add project root to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from rl.hierarchical_env import HierarchicalAttackEnv, AttackStrategy
from rl.hierarchical_agent_sac import HierarchicalAgentSAC, create_hierarchical_agent_sac
from models.victim_models import get_victim_model
from attacks import ATTACK_REGISTRY


# =============================================================================
# Dataset Configuration
# =============================================================================

MEDMNIST_DATASETS = {
    "pathmnist": {"n_classes": 9, "n_channels": 3},
    "dermamnist": {"n_classes": 7, "n_channels": 3},
    "bloodmnist": {"n_classes": 8, "n_channels": 3},
    "organamnist": {"n_classes": 11, "n_channels": 1},
    "organcmnist": {"n_classes": 11, "n_channels": 1},
    "organsmnist": {"n_classes": 11, "n_channels": 1},
}


def get_dataloader(
    dataset_name: str = "pathmnist", 
    batch_size: int = 1, 
    split: str = "train",
    native: bool = False
):
    """Get dataloader for training."""
    from data import get_dataset, get_dataloader as data_get_dataloader
    
    dataset_name = dataset_name.lower()
    
    if dataset_name not in MEDMNIST_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(MEDMNIST_DATASETS.keys())}")
    
    dataset = get_dataset(
        name=dataset_name,
        split=split,
        native=native,
        download=True
    )
    
    return data_get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=False
    )


# =============================================================================
# Training Metrics
# =============================================================================

class TrainingMetrics:
    """Track training metrics with rolling windows."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.successes = deque(maxlen=window_size)
        self.epsilons = deque(maxlen=window_size)
        self.strategy_counts = {s.name: 0 for s in AttackStrategy}
        
        # Per-strategy success tracking
        self.strategy_successes = {s.name: deque(maxlen=window_size) for s in AttackStrategy}
        self.strategy_rewards = {s.name: deque(maxlen=window_size) for s in AttackStrategy}
        
        # Totals
        self.total_episodes = 0
        self.total_successes = 0
        
        # History for trend analysis (store every N episodes)
        self.history_interval = 100  # Store metrics every 100 episodes
        self.history = {
            "episodes": [],
            "success_rate": [],
            "mean_reward": [],
            "mean_epsilon": [],
        }
        self._last_history_ep = 0
    
    def add_episode(
        self, 
        reward: float, 
        length: int, 
        success: bool, 
        strategy: int,
        epsilon: float = 0.0
    ):
        self.rewards.append(reward)
        self.episode_lengths.append(length)
        self.successes.append(float(success))
        self.epsilons.append(epsilon)
        
        strategy_name = AttackStrategy(strategy).name
        self.strategy_counts[strategy_name] += 1
        
        # Track per-strategy performance
        self.strategy_successes[strategy_name].append(float(success))
        self.strategy_rewards[strategy_name].append(reward)
        
        self.total_episodes += 1
        if success:
            self.total_successes += 1
        
        # Store history periodically
        if self.total_episodes - self._last_history_ep >= self.history_interval:
            self.history["episodes"].append(self.total_episodes)
            self.history["success_rate"].append(self.success_rate)
            self.history["mean_reward"].append(self.mean_reward)
            self.history["mean_epsilon"].append(self.mean_epsilon)
            self._last_history_ep = self.total_episodes
    
    def get_strategy_success_rates(self) -> Dict[str, float]:
        """Get success rate for each strategy."""
        rates = {}
        for name in self.strategy_successes:
            if len(self.strategy_successes[name]) > 0:
                rates[name] = np.mean(self.strategy_successes[name])
            else:
                rates[name] = 0.0
        return rates
    
    def get_strategy_rewards(self) -> Dict[str, float]:
        """Get mean reward for each strategy."""
        rewards = {}
        for name in self.strategy_rewards:
            if len(self.strategy_rewards[name]) > 0:
                rewards[name] = np.mean(self.strategy_rewards[name])
            else:
                rewards[name] = 0.0
        return rewards
    
    @property
    def mean_reward(self) -> float:
        return np.mean(self.rewards) if self.rewards else 0.0
    
    @property
    def mean_length(self) -> float:
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0
    
    @property
    def success_rate(self) -> float:
        return np.mean(self.successes) if self.successes else 0.0
    
    @property
    def mean_epsilon(self) -> float:
        return np.mean(self.epsilons) if self.epsilons else 0.0
    
    def get_strategy_distribution(self) -> str:
        total = sum(self.strategy_counts.values())
        if total == 0:
            return "N/A"
        parts = []
        for name, count in self.strategy_counts.items():
            pct = 100 * count / total
            if count > 0:
                parts.append(f"{name[0]}:{pct:.0f}%")
        return " ".join(parts)


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Hierarchical RL Agent with SAC Controller"
    )
    
    # Model and data
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to victim model checkpoint"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="pathmnist",
        choices=["pathmnist", "dermamnist", "bloodmnist", "organamnist"],
        help="MedMNIST dataset to use"
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="resnet18",
        help="Victim model architecture"
    )
    
    # Training
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use"
    )
    
    # SAC hyperparameters
    parser.add_argument(
        "--sac-lr",
        type=float,
        default=3e-4,
        help="SAC learning rate"
    )
    parser.add_argument(
        "--sac-buffer-size",
        type=int,
        default=100000,
        help="SAC replay buffer size"
    )
    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=256,
        help="SAC batch size"
    )
    parser.add_argument(
        "--sac-learning-starts",
        type=int,
        default=1000,
        help="Steps before SAC starts learning"
    )
    
    # Curriculum and budget
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning for strategies"
    )
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=0.1,
        help="Maximum epsilon budget (agent learns to use minimum needed, default: 0.1)"
    )
    parser.add_argument(
        "--balanced-warmup",
        type=int,
        default=2000,
        help="Number of episodes with forced round-robin strategy split before DQN takes over (default: 2000)"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["pixel", "frequency"],
        choices=["pixel", "frequency", "square"],
        help="Attack strategies to use (default: pixel frequency). Add 'square' for robust models."
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hierarchical_sac",
        help="Output directory"
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=2000,
        help="Logging frequency in steps (default: 2000)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Checkpoint save frequency (0 to disable)"
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="standard",
        choices=["standard", "robust"],
        help="Reward function mode: 'standard' for normal models, 'robust' for adversarially trained models (default: standard)"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # ==========================================================================
    # Print Configuration
    # ==========================================================================
    
    print("\n" + "="*70)
    print("  HIERARCHICAL RL TRAINING WITH SAC")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"    Victim model: {args.model_path}")
    print(f"    Dataset: {args.dataset}")
    print(f"    Device: {device}")
    print(f"    Total timesteps: {args.timesteps:,}")
    print(f"    Curriculum: {'Enabled' if args.curriculum else 'Disabled'}")
    print(f"    Target epsilon: {args.target_epsilon} (agent learns optimal)")
    print(f"    Strategies: {args.strategies}")
    print(f"\n  SAC Hyperparameters:")
    print(f"    Learning rate: {args.sac_lr}")
    print(f"    Buffer size: {args.sac_buffer_size:,}")
    print(f"    Batch size: {args.sac_batch_size}")
    print(f"    Learning starts: {args.sac_learning_starts}")
    print(f"    Entropy: auto-tuned")
    
    # ==========================================================================
    # Load Victim Model
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("  Loading victim model...")
    
    # Load checkpoint to get config
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Determine number of classes
    if isinstance(checkpoint, dict):
        if 'n_classes' in checkpoint:
            num_classes = checkpoint['n_classes']
        elif 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        elif 'model_config' in checkpoint:
            num_classes = checkpoint['model_config'].get('num_classes', 9)
        else:
            num_classes = 9
    else:
        num_classes = 9
    
    print(f"    Classes: {num_classes}")
    
    # Use get_victim_model which handles checkpoint loading
    victim_model = get_victim_model(
        architecture=args.model_arch,
        num_classes=num_classes,
        checkpoint=args.model_path,
        device=device
    )
    victim_model.eval()
    
    # Get native flag from checkpoint
    native = checkpoint.get('native', False)
    if native:
        print(f"    ✓ Native mode: 28x28 images")
    else:
        print(f"    ✓ Upscale mode: 224x224 images")
    
    # Get normalization stats (CRITICAL: must match evaluation)
    normalize_stats = checkpoint.get('normalize_stats', None)
    if normalize_stats is None:
        if native:
            n_channels = MEDMNIST_DATASETS.get(args.dataset, {}).get("n_channels", 3)
            if n_channels == 1:
                normalize_stats = {'mean': [0.5], 'std': [0.5]}
            else:
                normalize_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        else:
            normalize_stats = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
    print(f"    ✓ Normalize stats: mean={normalize_stats['mean']}, std={normalize_stats['std']}")
    
    print("    ✓ Model loaded successfully")
    
    # ==========================================================================
    # Load Data
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("  Loading data...")
    
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=1,
        native=native
    )
    print(f"    ✓ {len(train_loader.dataset)} training samples")
    
    # ==========================================================================
    # Create Environment
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("  Creating environment...")
    
    # Budget constants (must be defined before env_config)
    EPSILON_MIN = 0.001   # Minimum epsilon agent can use
    EPSILON_MAX = 0.1     # Maximum epsilon available - agent chooses optimal
    
    env_config = {
        "rl": {
            "action_space": {
                "epsilon_range": [EPSILON_MIN, args.target_epsilon]
            },
            "reward": {
                "success_reward": 1.0,
                "failure_penalty": -0.5
            }
        },
        "attack": {
            "max_iterations": 10
        }
    }
    
    env = HierarchicalAttackEnv(
        victim_model=victim_model,
        attack_registry=ATTACK_REGISTRY,
        dataloader=train_loader,
        config=env_config,
        device=device,
        mode="params_only",
        allowed_strategies=args.strategies,
        normalize_stats=normalize_stats,  # CRITICAL: ensures correct _clamp_image and epsilon interpretation
        reward_mode=args.reward_mode  # "standard" or "robust"
    )
    print(f"    ✓ Environment created (with normalize_stats)")
    print(f"    ✓ Reward mode: {args.reward_mode}")
    print(f"    ✓ Observation space: {env.observation_space.shape}")
    print(f"    ✓ Action space: {env.action_space.shape}")
    
    # ==========================================================================
    # Create Agent
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("  Creating SAC agent...")
    
    agent_config = {
        # SAC config
        "sac_lr": args.sac_lr,
        "sac_buffer_size": args.sac_buffer_size,
        "sac_batch_size": args.sac_batch_size,
        "sac_learning_starts": args.sac_learning_starts,
        "sac_tau": 0.005,
        "sac_gamma": 0.99,
        # Meta-controller (Custom DQN) config - stability focused
        "meta_lr": 5e-5,       # Lower LR
        "meta_buffer_size": 10000,
        "meta_batch_size": 32,
        "meta_gamma": 0.9,     # Lower gamma
        "meta_tau": 0.005,     # Slower soft update
        "meta_epsilon_start": 0.3,
        "meta_epsilon_min": 0.3,    # Fixed at 0.3 throughout training
        "meta_epsilon_decay": 1.0,  # No decay
    }
    
    agent = create_hierarchical_agent_sac(env, agent_config, device)
    
    print("    ✓ Agent created")
    
    # ==========================================================================
    # Curriculum Setup
    # ==========================================================================
    
    # Build strategy index list from args
    STRATEGY_NAME_TO_IDX = {"pixel": 0, "frequency": 1, "square": 2}
    strategy_indices = [STRATEGY_NAME_TO_IDX[s] for s in args.strategies]
    n_strategies = len(strategy_indices)
    
    # Curriculum phases (strategies only, budget is handled separately)
    curriculum_phases = [
        {
            "name": "Phase 1: Low epsilon only",
            "threshold": 0,
            "strategies": [strategy_indices[0]],
            "epsilon_min": args.target_epsilon * 0.3,  # 0.009 if target=0.03
            "epsilon_max": args.target_epsilon * 0.5,  # 0.015 if target=0.03
        },
        {
            "name": "Phase 2: Medium epsilon",
            "threshold": args.timesteps // 4,
            "strategies": strategy_indices,
            "epsilon_min": args.target_epsilon * 0.2,  # 0.006
            "epsilon_max": args.target_epsilon * 0.7,  # 0.021
        },
        {
            "name": "Phase 3: Full epsilon range",
            "threshold": args.timesteps // 2,
            "strategies": strategy_indices,
            "epsilon_min": EPSILON_MIN,  # 0.006 (20% of target)
            "epsilon_max": args.target_epsilon,  # 0.03
        },
    ]
    
    current_phase_idx = 0
    
    # Initialize strategies based on curriculum setting
    if args.curriculum:
        initial_strategies = curriculum_phases[0]["strategies"]
    else:
        initial_strategies = strategy_indices
    
    agent.set_allowed_strategies(initial_strategies)
    
    # Set initial epsilon range from curriculum if enabled
    if args.curriculum:
        initial_eps_min = curriculum_phases[0]["epsilon_min"]
        initial_eps_max = curriculum_phases[0]["epsilon_max"]
    else:
        initial_eps_min = EPSILON_MIN
        initial_eps_max = args.target_epsilon
    
    env.set_curriculum_config(
        epsilon_range=[initial_eps_min, initial_eps_max],
        curriculum_phase=0
    )
    
    print(f"\n  Initial Setup:")
    print(f"    ✓ Strategies: {[AttackStrategy(s).name for s in initial_strategies]}")
    print(f"    ✓ Epsilon range: [{initial_eps_min:.4f}, {initial_eps_max:.4f}]")
    
    if args.curriculum:
        print("\n  Curriculum Schedule:")
        for p in curriculum_phases:
            strategies = [AttackStrategy(s).name for s in p["strategies"]]
            print(f"    • {p['name']} (step {p['threshold']:,})")
            print(f"        Strategies: {strategies}")
            print(f"        Epsilon: [{p['epsilon_min']:.4f}, {p['epsilon_max']:.4f}]")
    
    print(f"\n  ⚖️  Balanced Warmup:")
    print(f"    • First {args.balanced_warmup} episodes: 50-50 strategy split (ignores DQN)")
    print(f"    • After warmup: DQN controls strategy selection")
    print(f"    • This ensures both SACs get enough training data")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    
    print("\n" + "="*70)
    print("  TRAINING")
    print("="*70 + "\n")
    
    metrics = TrainingMetrics(window_size=100)
    
    obs, info = env.reset()
    base_state = obs[:21]  # First 21 dims for meta-controller (excludes strategy one-hot)
    
    # Warmup: alternate strategies 50-50, after warmup: use DQN
    warmup_episodes = args.balanced_warmup
    episode_counter = 0
    
    if episode_counter < warmup_episodes:
        # Balanced warmup: alternate between strategies
        current_strategy = episode_counter % agent.n_strategies
    else:
        current_strategy = agent.meta_controller.select_strategy(base_state)
    
    obs = env.set_strategy(current_strategy)  # Returns updated obs with strategy one-hot
    obs = np.array(obs, dtype=np.float32).flatten()  # Ensure correct format for SAC
    strategy_start_state = base_state.copy()
    
    episode_reward = 0
    episode_steps = 0
    episode_success = False
    episode_epsilon = 0.0
    
    # Progress bar
    pbar = tqdm(
        range(args.timesteps),
        desc="Training",
        unit="step",
        ncols=140,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    )
    pbar.set_postfix({
        "ph": 1, "ep": 0, "rew": "0.0", "sr": "0%", 
        "bgt": f"{args.target_epsilon:.2f}", "ε_cur": "0.000", "ε_avg": "0.000",
        "str": "N/A", "ent": "?"
    })
    
    # Training loop
    for step in pbar:
        # Curriculum update
        if args.curriculum:
            while (current_phase_idx < len(curriculum_phases) - 1 and 
                   step >= curriculum_phases[current_phase_idx + 1]["threshold"]):
                current_phase_idx += 1
                new_phase = curriculum_phases[current_phase_idx]
                agent.set_allowed_strategies(new_phase["strategies"])
                
                env.set_curriculum_config(
                    epsilon_range=[new_phase["epsilon_min"], new_phase["epsilon_max"]],
                    curriculum_phase=current_phase_idx
                )
                
                tqdm.write(f"\n{'='*55}")
                tqdm.write(f"  {new_phase['name']}")
                tqdm.write(f"  Strategies: {[AttackStrategy(s).name for s in new_phase['strategies']]}")
                tqdm.write(f"  Epsilon range: [{new_phase['epsilon_min']:.4f}, {new_phase['epsilon_max']:.4f}]")
                tqdm.write(f"{'='*55}\n")
        
        # Controller selects parameters (using the strategy-specific SAC)
        # obs is already a 1D float32 numpy array
        # Con SAC separati, usiamo solo base_state (primi 21 dims)
        base_state = obs[:21]
        params, _ = agent.controllers[current_strategy].predict(base_state, deterministic=False)
        params = np.asarray(params).flatten()
        
        # =================================================================
        # FORCED EXPLORATION: Override actions with random values
        # Decaying probability: starts high, decreases over training
        # This forces the agent to explore different epsilon values
        # =================================================================
        exploration_rate = max(0.05, 0.5 * (1 - step / args.timesteps))  # 50% → 5%
        if np.random.random() < exploration_rate:
            # Random action in [0, 1] for all parameters
            params = np.random.uniform(0, 1, size=params.shape).astype(np.float32)
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(params)
        done = terminated or truncated
        
        # Add to the SPECIFIC strategy's replay buffer
        next_base_state = np.array(next_obs[:21], dtype=np.float32)
        agent.add_to_buffer(
            strategy=current_strategy,
            state=base_state,
            action=params,
            reward=reward,
            next_state=next_base_state,
            done=done
        )
        
        episode_reward += reward
        episode_steps += 1
        agent.total_timesteps = step + 1
        
        # Track epsilon
        episode_epsilon = info.get("params", {}).get("epsilon", 0.0)
        
        if info.get("success", False):
            episode_success = True
        
        # Train ALL SACs that have enough data (not just the active one!)
        # Each SAC has its own replay buffer - training from buffer doesn't
        # require new env data, just gradient steps on existing transitions.
        # This prevents the "death spiral" where the unused SAC stagnates.
        if step >= args.sac_learning_starts and step % 4 == 0:
            if step == args.sac_learning_starts:
                tqdm.write(f"\n🚀 SAC training started at step {step}\n")
            for strategy_idx in range(agent.n_strategies):
                agent.train_controller(strategy_idx, gradient_steps=1)
        
        # Periodic SAC logging (every log_freq steps)
        if step > 0 and step % args.log_freq == 0:
            try:
                # Get SAC metrics for each controller
                controller_stats = agent.get_controller_stats()
                
                # Get MetaController (DQN) metrics
                meta_stats = agent.meta_controller.get_stats()
                
                # Get per-strategy success rates
                strategy_sr = metrics.get_strategy_success_rates()
                strategy_rew = metrics.get_strategy_rewards()
                
                # Get Q-values for current state (sample)
                try:
                    q_values = agent.meta_controller.get_q_values(obs[:21])
                    def fmt_q(v):
                        if abs(v) > 1000:
                            return f"{v:.1e}"
                        return f"{v:.2f}"
                    strategy_names = ["P", "F", "S"][:len(q_values)]
                    q_parts = [f"{name}:{fmt_q(q_values[i])}" for i, name in enumerate(strategy_names)]
                    q_str = " ".join(q_parts)
                except Exception as e:
                    q_str = f"N/A ({e})"
                
                # Calculate trend
                trend_str = ""
                if len(metrics.history["success_rate"]) >= 2:
                    prev_sr = metrics.history["success_rate"][-2]
                    curr_sr = metrics.success_rate
                    diff = (curr_sr - prev_sr) * 100
                    if diff > 0:
                        trend_str = f" (↑ +{diff:.1f}%)"
                    elif diff < 0:
                        trend_str = f" (↓ {diff:.1f}%)"
                    else:
                        trend_str = " (→)"
                
                # Log learning status
                log_msg = f"\n{'═'*65}\n"
                log_msg += f"  📊 Step {step:,} | Episodes: {metrics.total_episodes}\n"
                log_msg += f"{'═'*65}\n"
                
                # Overall performance
                log_msg += f"  OVERALL: SR={metrics.success_rate*100:.0f}%{trend_str} | "
                log_msg += f"Rew={metrics.mean_reward:.2f} | ε_attack={metrics.mean_epsilon:.4f}\n"
                
                # Warmup status
                if episode_counter < warmup_episodes:
                    log_msg += f"  ⚖️  WARMUP MODE: {episode_counter}/{warmup_episodes} episodes (50-50 split)\n"
                else:
                    log_msg += f"  🎯 DQN MODE: DQN controls strategy selection\n"
                log_msg += f"{'─'*65}\n"
                
                # Per-strategy performance with SAC buffer info
                log_msg += f"  STRATEGY PERFORMANCE (last {metrics.window_size} episodes each):\n"
                for name in [s.upper() for s in args.strategies]:
                    sr = strategy_sr.get(name, 0) * 100
                    rew = strategy_rew.get(name, 0)
                    count_pct = metrics.strategy_counts.get(name, 0) / max(1, metrics.total_episodes) * 100
                    # Get buffer size for this strategy's SAC
                    sac_stats = controller_stats.get(name, {})
                    buf_size = sac_stats.get('buffer_size', 0)
                    log_msg += f"    {name:10s}: SR={sr:5.1f}% | Rew={rew:5.2f} | Usage={count_pct:4.1f}% | Buf={buf_size:,}\n"
                
                log_msg += f"{'─'*65}\n"
                
                # DQN learning
                log_msg += f"  DQN Meta-Controller:\n"
                log_msg += f"    ε={meta_stats['epsilon']:.3f} | Loss={meta_stats.get('mean_loss', 0):.4f}\n"
                log_msg += f"    Q-values: [{q_str}]\n"
                log_msg += f"    Buffer={meta_stats['buffer_size']:,} | Updates={meta_stats['total_updates']:,}\n"
                
                # SAC controllers summary
                log_msg += f"  SAC Controllers (separate per strategy):\n"
                for name, stats in controller_stats.items():
                    log_msg += f"    SAC_{name}: Buffer={stats['buffer_size']:,} | Ent={stats['ent_coef']:.2f}\n"
                log_msg += f"{'═'*65}"
                tqdm.write(log_msg)
            except Exception as e:
                tqdm.write(f"[Logging] step={step}, error: {e}")
        
        if done:
            # Update meta-controller
            next_base_state = next_obs[:21]  # Base state for meta-controller
            agent.meta_controller.store_transition(
                strategy_start_state,
                current_strategy,
                episode_reward,
                next_base_state,
                True
            )
            agent.meta_controller.update()
            
            # Track strategy usage
            actual_strategy = info.get("strategy", "PIXEL")
            strategy_idx = {"PIXEL": 0, "FREQUENCY": 1, "SQUARE": 2}.get(actual_strategy, 0)
            
            # Update metrics
            metrics.add_episode(episode_reward, episode_steps, episode_success, strategy_idx, episode_epsilon)
            agent.episode_count = metrics.total_episodes
            agent.strategy_usage[AttackStrategy(strategy_idx).name] += 1
            
            # Reset
            obs, info = env.reset()
            base_state = obs[:21]
            
            # Warmup: alternate strategies 50-50, after warmup: use DQN
            episode_counter += 1
            if episode_counter < warmup_episodes:
                # Balanced warmup: alternate between strategies
                current_strategy = episode_counter % agent.n_strategies
            else:
                # Log when warmup ends
                if episode_counter == warmup_episodes:
                    tqdm.write(f"\n{'='*55}")
                    tqdm.write(f"  🎯 Balanced warmup complete ({warmup_episodes} episodes)")
                    tqdm.write(f"  DQN now controls strategy selection")
                    controller_stats = agent.get_controller_stats()
                    for name, stats in controller_stats.items():
                        tqdm.write(f"    SAC_{name}: Buffer={stats['buffer_size']:,}")
                    tqdm.write(f"{'='*55}\n")
                current_strategy = agent.meta_controller.select_strategy(base_state)
            
            obs = env.set_strategy(current_strategy)  # Get updated obs with strategy
            obs = np.array(obs, dtype=np.float32).flatten()  # Ensure correct format
            strategy_start_state = base_state.copy()
            
            episode_reward = 0
            episode_steps = 0
            episode_success = False
        else:
            obs = np.array(next_obs, dtype=np.float32).flatten()  # Ensure correct format
            base_state = obs[:21]
        
        # Update progress bar
        if step % 100 == 0:
            current_budget = env.epsilon_range[1] if hasattr(env, 'epsilon_range') else args.target_epsilon
            
            # Get entropy coefficient from current strategy's SAC
            try:
                ent_val = float(agent.controllers[current_strategy].ent_coef)
                ent_str = f"{ent_val:.2f}"
            except:
                try:
                    ent_val = agent.controllers[current_strategy].log_ent_coef.exp().item()
                    ent_str = f"{ent_val:.2f}"
                except:
                    ent_str = "?"
            
            # Get DQN epsilon
            dqn_eps = agent.meta_controller.epsilon
            
            # Show warmup status
            warmup_str = "WU" if episode_counter < warmup_episodes else "DQN"
            
            pbar.set_postfix(ordered_dict={
                "ep": metrics.total_episodes,
                "rew": f"{metrics.mean_reward:.2f}",
                "sr": f"{metrics.success_rate*100:.0f}%",
                "ε_atk": f"{metrics.mean_epsilon:.3f}",
                "mode": warmup_str,
                "str": metrics.get_strategy_distribution(),
            }, refresh=True)
        
        # Save checkpoint
        if args.save_freq > 0 and step > 0 and step % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_{step}"
            agent.save(str(checkpoint_path))
            # Save state extractor statistics
            env.state_extractor.save_stats(str(checkpoint_path / "state_extractor_stats.pt"))
            tqdm.write(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # ==========================================================================
    # Save Final Model
    # ==========================================================================
    
    pbar.close()
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    
    final_path = output_dir / "final_agent"
    agent.save(str(final_path))
    
    # Save state extractor statistics
    state_stats_path = final_path / "state_extractor_stats.pt"
    env.state_extractor.save_stats(str(state_stats_path))
    print(f"  State extractor stats saved to: {state_stats_path}")
    
    print(f"\n  Results:")
    print(f"    Total episodes: {metrics.total_episodes:,}")
    print(f"    Final success rate: {metrics.success_rate*100:.1f}%")
    print(f"    Final mean reward: {metrics.mean_reward:.2f}")
    print(f"    Mean epsilon used: {metrics.mean_epsilon:.4f}")
    print(f"\n  Strategy usage:")
    for name, count in agent.strategy_usage.items():
        pct = 100 * count / max(1, metrics.total_episodes)
        print(f"    {name}: {count:,} ({pct:.1f}%)")
    
    # DQN Meta-Controller stats
    meta_stats = agent.meta_controller.get_stats()
    print(f"\n  DQN Meta-Controller:")
    print(f"    Final epsilon: {meta_stats['epsilon']:.3f}")
    print(f"    Buffer size:   {meta_stats['buffer_size']:,}")
    print(f"    Total updates: {meta_stats['total_updates']:,}")
    
    print(f"\n  Model saved to: {final_path}")
    
    # Save training log
    log = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "final_metrics": {
            "total_episodes": metrics.total_episodes,
            "success_rate": metrics.success_rate,
            "mean_reward": metrics.mean_reward,
            "mean_epsilon": metrics.mean_epsilon,
        },
        "strategy_usage": agent.strategy_usage,
        "training_history": metrics.history,  # For plotting learning curves
        "dqn_meta_controller": meta_stats,  # DQN stats
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"  Training log saved to: {output_dir / 'training_log.json'}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()