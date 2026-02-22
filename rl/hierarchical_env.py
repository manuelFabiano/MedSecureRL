"""
Hierarchical Environment for Adversarial Attack Generation.

Separates strategy selection (discrete) from parameter optimization (continuous).
This allows cleaner learning with appropriate algorithms for each level.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum

from .state_extractor import StateExtractor
from .reward import RewardFunction, get_reward_function


class AttackStrategy(IntEnum):
    """Available attack strategies."""
    PIXEL = 0
    FREQUENCY = 1
    SQUARE = 2



@dataclass
class StrategyConfig:
    """Configuration for a specific strategy's parameter space."""
    name: str
    param_names: List[str]
    param_ranges: Dict[str, Tuple[float, float]]
    param_dim: int


# Define parameter spaces for each strategy
STRATEGY_CONFIGS = {
    AttackStrategy.PIXEL: StrategyConfig(
        name="pixel",
        param_names=["epsilon", "iterations", "step_size", "momentum"],
        param_ranges={
            "epsilon": (0.001, 0.1),
            "iterations": (1, 100),
            "step_size": (0.001, 0.1),
            "momentum": (0.0, 0.95),
        },
        param_dim=4
    ),
    AttackStrategy.FREQUENCY: StrategyConfig(
        name="frequency",
        param_names=["epsilon", "iterations", "step_size", "band", "band_ratio"],
        param_ranges={
            "epsilon": (0.001, 0.1),
            "iterations": (1, 100),
            "step_size": (0.001, 0.1),
            "band": (0.0, 3.99),         # 0=low, 1=mid, 2=high, 3=adaptive
            "band_ratio": (0.3, 0.6),    # INCREASED: was (0.1, 0.5) - low values kill gradient
        },
        param_dim=5
    ),
    AttackStrategy.SQUARE: StrategyConfig(
        name="square",
        param_names=["epsilon", "p_init"],
        param_ranges={
            "epsilon": (0.001, 0.1),
            "p_init": (0.05, 0.8),
        },
        param_dim=2
    ),
}

# Maximum parameter dimension across all strategies
MAX_PARAM_DIM = max(cfg.param_dim for cfg in STRATEGY_CONFIGS.values())


class HierarchicalAttackEnv(gym.Env):
    """
    Hierarchical environment for adversarial attacks.
    
    Two-level action space:
    1. Strategy selection (discrete): Which attack type to use
    2. Parameter selection (continuous): How to configure that attack
    
    The environment can be used in two modes:
    - Combined: Single agent outputs both strategy and parameters
    - Separated: External meta-controller selects strategy, agent selects parameters
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        victim_model,
        attack_registry: Dict[str, Any],
        dataloader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        mode: str = "combined",  # "combined" or "separated"
        allowed_strategies: Optional[List[str]] = None,
        normalize_stats: Optional[Dict[str, List[float]]] = None,
        reward_mode: str = "standard",  # "standard" or "robust"
    ):
        """
        Initialize hierarchical environment.
        
        Args:
            victim_model: Target model to attack
            attack_registry: Dictionary of attack classes
            dataloader: DataLoader for images
            config: Configuration dictionary
            device: Compute device
            mode: "combined" (single agent) or "separated" (hierarchical agents)
            allowed_strategies: List of allowed strategy names
            normalize_stats: Dict with 'mean' and 'std' for image normalization
        """
        super().__init__()
        
        self.victim_model = victim_model
        self.attack_registry = attack_registry
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.config = config
        self.device = device
        self.mode = mode
        
        # Normalization stats for attacks
        self.normalize_stats = normalize_stats
        
        # Set allowed strategies
        if allowed_strategies:
            self.allowed_strategies = [
                AttackStrategy[s.upper()] for s in allowed_strategies
            ]
        else:
            self.allowed_strategies = list(AttackStrategy)
        
        self.n_strategies = len(self.allowed_strategies)
        
        # State extractor
        self.state_extractor = StateExtractor(victim_model, device, max_steps=config.get("max_steps_per_image", 5))
        self.state_dim = self.state_extractor.STATE_DIM  # 21 (includes epsilon_budget)
        
        # Number of strategies determines one-hot encoding size
        self.n_strategies = len(self.allowed_strategies)
        
        # Define observation space
        # State (21) + one-hot strategy (n_strategies) for parameter agent
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.state_dim + self.n_strategies,),  # 21 + n_strategies
            dtype=np.float32
        )
        
        # Define action space based on mode
        # NOTE: Strategy space uses actual number of allowed strategies
        if mode == "combined":
            # Combined: Dict space with strategy (discrete) and params (continuous)
            self.action_space = spaces.Dict({
                "strategy": spaces.Discrete(self.n_strategies),
                "params": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_PARAM_DIM,),
                    dtype=np.float32
                )
            })
        else:
            # Separated: Only continuous parameters (strategy set externally)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(MAX_PARAM_DIM,),
                dtype=np.float32
            )
        
        # Episode state
        self.current_image = None
        self.current_label = None
        self.current_adversarial = None
        self.current_strategy = None
        self.current_epsilon = 0.0  # Track epsilon used in current/last attack
        self.episode_step = 0
        self.max_steps = config.get("max_steps_per_image", 5)
        
        # Curriculum config (can be updated externally)
        # Wide epsilon range: agent learns to find optimal trade-off
        self.epsilon_range = config.get("rl", {}).get("action_space", {}).get(
            "epsilon_range", [0.001, 0.1]  # Wide range for adaptive selection
        )
        
        # Attack cache
        self._attacks = {}
        
        # Metrics
        self.total_attacks = 0
        self.successful_attacks = 0
        
        # Multi-objective reward function
        self.reward_mode = reward_mode
        self.reward_fn = get_reward_function(config, device, normalize_stats, mode=reward_mode)
        self.curriculum_phase = 0
    
    def set_strategy(self, strategy: int) -> np.ndarray:
        """Set strategy externally (for separated mode).
        
        Returns:
            Updated observation with new strategy one-hot encoding.
        """
        if strategy < len(self.allowed_strategies):
            self.current_strategy = self.allowed_strategies[strategy]
        else:
            self.current_strategy = self.allowed_strategies[0]
        
        # Return updated observation with new strategy
        return self._compute_observation()
    
    def set_curriculum_config(
        self,
        allowed_strategies: List[str] = None,
        epsilon_range: Tuple[float, float] = None,
        curriculum_phase: int = None,
        **kwargs
    ):
        """Update curriculum configuration."""
        if allowed_strategies:
            self.allowed_strategies = [
                AttackStrategy[s.upper()] for s in allowed_strategies
                if s.upper() in AttackStrategy.__members__
            ]
            self.n_strategies = len(self.allowed_strategies)
        
        if epsilon_range:
            self.epsilon_range = list(epsilon_range)
        
        if curriculum_phase is not None:
            self.curriculum_phase = curriculum_phase
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Optional dict with:
                - 'image': torch.Tensor [1, C, H, W] to use instead of dataloader
                - 'label': torch.Tensor [1] corresponding label
        """
        super().reset(seed=seed)
        
        # Check if specific image provided via options
        if options is not None and 'image' in options and 'label' in options:
            self.current_image = options['image'].to(self.device)
            self.current_label = options['label'].to(self.device)
        else:
            # Get new image from dataloader
            self.current_image, self.current_label = self._get_next_image()
        
        self.current_adversarial = None
        self.current_strategy = None
        self.current_epsilon = 0.0  # Reset epsilon tracking
        self.episode_step = 0
        
        # Reset cached image features (new image = new features)
        self.state_extractor.reset_cache()
        
        # Compute observation
        obs = self._compute_observation()
        
        info = {
            "image_shape": tuple(self.current_image.shape),
            "true_label": self.current_label.item(),
            "allowed_strategies": [s.name for s in self.allowed_strategies],
        }
        
        return obs, info
    
    def step(
        self,
        action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        self.episode_step += 1
        
        # Decode action based on mode
        if self.mode == "combined":
            strategy_idx = action["strategy"]
            params = action["params"]
            
            # strategy_idx is the actual strategy enum value (0=PIXEL, 1=FREQ)
            strategy = AttackStrategy(strategy_idx)
            
            # Validate it's allowed
            if strategy in self.allowed_strategies:
                self.current_strategy = strategy
            else:
                # This should not happen if MetaController is properly configured
                # Log warning and fallback to first allowed
                print(f"WARNING: Strategy {strategy.name} not in allowed {[s.name for s in self.allowed_strategies]}. "
                      f"Falling back to {self.allowed_strategies[0].name}")
                self.current_strategy = self.allowed_strategies[0]
        else:
            # Strategy should be set externally via set_strategy()
            if self.current_strategy is None:
                self.current_strategy = self.allowed_strategies[0]
            params = action
        
        # Decode parameters for current strategy
        attack_params = self._decode_params(self.current_strategy, params)
        
        # Track epsilon used for observation
        self.current_epsilon = attack_params.get("epsilon", 0.0)
        
        # Execute attack
        result = self._execute_attack(self.current_strategy, attack_params)
        
        # Update adversarial
        if result.get("adversarial") is not None:
            self.current_adversarial = result["adversarial"]
        
        # Compute reward (returns reward and quality-based success flag)
        reward, quality_success = self._compute_reward(result, attack_params)
        
        # Check termination - use quality_success (requires SSIM >= 0.90)
        # This teaches agent that low-quality attacks are failures
        terminated = quality_success
        truncated = self.episode_step >= self.max_steps
        
        # Square Attack: always terminate after 1 step (2000 queries = full attack)
        if self.current_strategy == AttackStrategy.SQUARE:
            truncated = True
        
        # Metrics - track both raw misclassification and quality success
        self.total_attacks += 1
        if quality_success:
            self.successful_attacks += 1
        
        # New observation
        obs = self._compute_observation()
        
        info = {
            "success": quality_success,  # Quality-based success
            "misclassified": result["success"],  # Raw misclassification
            "strategy": self.current_strategy.name,
            "params": attack_params,
            "queries": result.get("queries", 0),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_observation(self) -> np.ndarray:
        """Compute observation with strategy embedding."""
        # Base state from state extractor
        budget = self.epsilon_range[1]  # Max allowed epsilon
        state = self.state_extractor.extract(
            x=self.current_image,
            y=self.current_label,
            x_adv=self.current_adversarial,
            attack_iteration=self.episode_step,
            current_strategy=self.current_strategy.value if self.current_strategy else 0,
            epsilon=self.current_epsilon if self.current_epsilon > 0 else budget,
            epsilon_budget=budget,
        )
        state = state.detach().cpu().numpy().flatten()
        
        # One-hot strategy embedding (relative to allowed_strategies)
        strategy_onehot = np.zeros(self.n_strategies, dtype=np.float32)
        if self.current_strategy is not None:
            # Find index in allowed_strategies list
            try:
                strategy_idx = self.allowed_strategies.index(self.current_strategy)
                strategy_onehot[strategy_idx] = 1.0
            except ValueError:
                pass  # Strategy not in allowed list, leave as zeros
        
        # Concatenate
        obs = np.concatenate([state, strategy_onehot])
        
        return obs.astype(np.float32)
    
    def _decode_params(
        self,
        strategy: AttackStrategy,
        params: np.ndarray
    ) -> Dict[str, Any]:
        """Decode continuous parameters for a strategy."""
        config = STRATEGY_CONFIGS[strategy]
        decoded = {}
        
        for i, name in enumerate(config.param_names):
            if i >= len(params):
                break
            
            # params[i] ∈ [0, 1] (SB3 SAC rescales to action_space range automatically)
            param_val = float(params[i])
            
            if name == "epsilon":
                # Map [0, 1] to epsilon range
                min_eps = self.epsilon_range[0]
                max_eps = self.epsilon_range[1]
                value = min_eps + param_val * (max_eps - min_eps)
            else:
                low, high = config.param_ranges[name]
                value = low + param_val * (high - low)
            
            # Special handling for discrete params
            if name in ["iterations", "n_queries"]:
                value = int(value)
            elif name == "band":
                value = int(value)
            
            decoded[name] = value
        
        return decoded
    
    def _execute_attack(
        self,
        strategy: AttackStrategy,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute attack with given strategy and parameters."""
        # Get attack class
        strategy_name = strategy.name.lower()
        if strategy_name not in self.attack_registry:
            return {"success": False, "adversarial": self.current_image}
        
        if strategy not in self._attacks:
            attack_class = self.attack_registry[strategy_name]
            # Pass normalize_stats to attack
            self._attacks[strategy] = attack_class(
                model=self.victim_model,
                device=self.device,
                normalize_stats=self.normalize_stats
            )
        
        attack = self._attacks[strategy]
        
        # Build kwargs based on strategy
        kwargs = {
            "epsilon": params.get("epsilon", 0.03),
            "iterations": params.get("iterations", 40),
            "step_size": params.get("step_size", 0.01),
        }
        
        if strategy == AttackStrategy.PIXEL:
            kwargs["momentum"] = params.get("momentum", 0.0)
            # Safe defaults for removed params (no longer agent-controlled)
            kwargs["targeted"] = False
            kwargs["random_start"] = True
            kwargs["sparsity"] = 1.0
            
        elif strategy == AttackStrategy.FREQUENCY:
            band_names = ["low", "mid", "high", "adaptive"]
            band_idx = int(params.get("band", 0))
            kwargs["target_bands"] = band_names[min(band_idx, 3)]
            kwargs["band_ratio"] = params.get("band_ratio", 0.4)  # Default 0.4
        
        elif strategy == AttackStrategy.SQUARE:
            # Square Attack: fixed query budget (not learnable - too expensive for RL)
            # Agent only controls epsilon and p_init
            kwargs = {
                "epsilon": params.get("epsilon", 0.03),
                "n_queries": 2000,  # Fixed: enough for 224x224, not too slow for training
                "p_init": params.get("p_init", 0.8),
            }

        # Execute
        try:
            result = attack.attack(
                x=self.current_image,
                y=self.current_label,
                **kwargs
            )
            
            return {
                "success": result.success if isinstance(result.success, bool) else result.success.any().item(),
                "adversarial": result.adversarial,
                "perturbation": result.perturbation,
                "queries": attack.query_count
            }
        except Exception as e:
            print(f"Attack error: {e}")
            return {"success": False, "adversarial": self.current_image}
    
    def _compute_reward(
        self,
        result: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Compute multi-objective reward using RewardFunction.
        
        Returns:
            Tuple of (reward, quality_success) where quality_success is True
            only if attack succeeded AND SSIM >= 0.90
        """
        # Get adversarial image (or original if attack failed)
        adversarial = result.get("adversarial", self.current_image)
        if adversarial is None:
            adversarial = self.current_image
        
        # Get model output on adversarial
        with torch.no_grad():
            model_output = self.victim_model(adversarial)
        
        # Build attack_info dict for reward function
        attack_info = {
            "success": result.get("success", False),
            "strategy": self.current_strategy.name if self.current_strategy else "PIXEL",
            "epsilon": params.get("epsilon", 0.03),
            "epsilon_budget": self.epsilon_range[1],  # Actual current budget
            "epsilon_min": self.epsilon_range[0],     # Min epsilon for efficiency bonus
            "epsilon_max": self.epsilon_range[1],     # Max epsilon for efficiency bonus
            "iterations": params.get("iterations", 40),
            "queries": result.get("queries", 0),
            "max_iterations": 100,
            "step": self.episode_step,
        }
        
        # Compute multi-objective reward
        reward_components = self.reward_fn.compute_reward(
            original=self.current_image,
            adversarial=adversarial,
            labels=self.current_label,
            model_output=model_output,
            attack_info=attack_info,
            curriculum_phase=self.curriculum_phase
        )
        
        # attack_info is modified by compute_reward to include quality-based success
        quality_success = attack_info.get("success", False)
        
        return reward_components.total, quality_success
    
    def _get_next_image(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next image from dataloader."""
        try:
            images, labels = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            images, labels = next(self.data_iterator)
        
        image = images[0:1].to(self.device)
        label = labels[0:1].to(self.device)
        
        if label.dim() > 1:
            label = label.squeeze()
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        return image, label


class MetaControllerEnv(gym.Env):
    """
    Environment for the meta-controller (strategy selection).
    
    This is a simple discrete action space environment that:
    - Observes image features
    - Outputs strategy selection
    - Gets reward based on overall episode success
    """
    
    def __init__(
        self,
        victim_model,
        dataloader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        allowed_strategies: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.victim_model = victim_model
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.config = config
        self.device = device
        
        # Strategies
        if allowed_strategies:
            self.allowed_strategies = [
                AttackStrategy[s.upper()] for s in allowed_strategies
            ]
        else:
            self.allowed_strategies = list(AttackStrategy)
        
        # State extractor
        self.state_extractor = StateExtractor(victim_model, device, max_steps=config.get("max_steps_per_image", 5))
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.state_extractor.STATE_DIM,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.allowed_strategies))
        
        # State
        self.current_image = None
        self.current_label = None
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            images, labels = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            images, labels = next(self.data_iterator)
        
        self.current_image = images[0:1].to(self.device)
        self.current_label = labels[0:1].to(self.device)
        
        state = self.state_extractor.extract(
            x=self.current_image,
            y=self.current_label,
            epsilon_budget=0.03  # Default budget for meta-controller
        )
        
        return state.detach().cpu().numpy().flatten(), {}
    
    def step(self, action):
        # This env is typically used just for strategy selection
        # The actual attack execution happens in HierarchicalAttackEnv
        strategy = self.allowed_strategies[action]
        
        # Return dummy values - real reward comes from lower-level
        return np.zeros(self.observation_space.shape), 0.0, True, False, {
            "selected_strategy": strategy.name
        }
    
    def set_curriculum_config(self, allowed_strategies=None, **kwargs):
        if allowed_strategies:
            self.allowed_strategies = [
                AttackStrategy[s.upper()] for s in allowed_strategies
                if s.upper() in AttackStrategy.__members__
            ]