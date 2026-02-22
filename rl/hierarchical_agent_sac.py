"""
Hierarchical RL Agent with SEPARATE SAC Controllers per Strategy.

Architecture:
- Meta-Controller (Custom DQN): Selects attack strategy
- Controllers (SAC per strategy): Each strategy has its own SAC

This eliminates interference between strategies - each SAC learns
optimal parameters only for its strategy.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure




# =============================================================================
# Simple Q-Network for Meta-Controller
# =============================================================================

class SimpleQNetwork(torch.nn.Module):
    """Simple Q-Network with layer normalization for stability."""
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_actions)
        )
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=0.5)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Meta-Controller (Strategy Selection) - Custom DQN
# =============================================================================

class MetaController:
    """Meta-Controller for strategy selection using custom DQN."""
    
    def __init__(
        self,
        state_dim: int,
        n_strategies: int = 2,
        config: Dict[str, Any] = None,
        device: str = "cuda"
    ):
        self.state_dim = state_dim
        self.n_strategies = n_strategies
        self.device = torch.device(device)
        config = config or {}
        
        # Networks
        self.q_net = SimpleQNetwork(state_dim, n_strategies).to(self.device)
        self.target_net = SimpleQNetwork(state_dim, n_strategies).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.lr = config.get("meta_lr", 5e-5)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Hyperparameters
        self.gamma = config.get("meta_gamma", 0.9)
        self.tau = config.get("meta_tau", 0.005)
        self.batch_size = config.get("meta_batch_size", 32)
        
        # Exploration
        self.epsilon = config.get("meta_epsilon_start", 1.0)
        self.epsilon_min = config.get("meta_epsilon_min", 0.05)
        self.epsilon_decay = config.get("meta_epsilon_decay", 0.999)
        
        # Replay buffer
        self.buffer_size = config.get("meta_buffer_size", 10000)
        self.buffer = []
        self.buffer_idx = 0
        
        # State normalization
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_std = np.ones(state_dim, dtype=np.float32)
        self.state_count = 0
        
        # Allowed strategies
        self.allowed_strategies = list(range(n_strategies))
        
        # Stats
        self.total_updates = 0
        self.episode_count = 0
        self.losses = []
        
        print(f"[DQN Meta-Controller] Initialized:")
        print(f"      - State dim: {state_dim}, Strategies: {n_strategies}")
        print(f"      - LR: {self.lr}, Gamma: {self.gamma}, Tau: {self.tau}")
    
    def _normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize state using running statistics."""
        state = np.array(state, dtype=np.float32).flatten()
        state = np.clip(state, -10, 10)
        
        if update_stats and self.state_count < 10000:
            self.state_count += 1
            delta = state - self.state_mean
            self.state_mean += delta / self.state_count
            self.state_std = np.sqrt(
                ((self.state_count - 1) * self.state_std**2 + delta * (state - self.state_mean)) 
                / self.state_count
            )
            self.state_std = np.maximum(self.state_std, 1e-6)
        
        normalized = (state - self.state_mean) / (self.state_std + 1e-6)
        return np.clip(normalized, -5, 5)
    
    def select_strategy(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select strategy using epsilon-greedy."""
        state_norm = self._normalize_state(state, update_stats=not deterministic)
        
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.choice(self.allowed_strategies)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).squeeze().cpu().numpy()
        
        masked_q = np.full(self.n_strategies, -np.inf)
        for s in self.allowed_strategies:
            masked_q[s] = q_values[s]
        
        return int(np.argmax(masked_q))
    
    def store_transition(self, state, strategy, reward, next_state, done):
        """Store transition."""
        state_norm = self._normalize_state(state, update_stats=True)
        next_state_norm = self._normalize_state(next_state, update_stats=True)
        clipped_reward = np.clip(reward, -2.0, 2.0)
        
        transition = (state_norm, strategy, clipped_reward, next_state_norm, done)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.buffer_idx] = transition
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update(self) -> Optional[float]:
        """Update Q-network with Double DQN."""
        if len(self.buffer) < self.batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.total_updates += 1
        loss_val = loss.item()
        self.losses.append(loss_val)
        
        return loss_val
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "total_updates": self.total_updates,
            "episode_count": self.episode_count,
            "mean_loss": np.mean(self.losses[-100:]) if self.losses else 0.0,
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_norm = self._normalize_state(state, update_stats=False)
        with torch.no_grad():
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            return self.q_net(state_t).squeeze().cpu().numpy()
    
    def set_allowed_strategies(self, strategies: List[int]):
        self.allowed_strategies = strategies
    
    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episode_count': self.episode_count,
            'allowed_strategies': self.allowed_strategies,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_count': self.state_count,
        }, f"{path}_dqn.pt")
    
    def load(self, path: str):
        checkpoint = torch.load(f"{path}_dqn.pt", map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_updates = checkpoint['total_updates']
        self.episode_count = checkpoint['episode_count']
        self.allowed_strategies = checkpoint.get('allowed_strategies', list(range(self.n_strategies)))
        self.state_mean = checkpoint.get('state_mean', np.zeros(self.state_dim, dtype=np.float32))
        self.state_std = checkpoint.get('state_std', np.ones(self.state_dim, dtype=np.float32))
        self.state_count = checkpoint.get('state_count', 0)


# =============================================================================
# Dummy Environment for Individual SAC Controllers
# =============================================================================

class SingleStrategySACEnv(gym.Env):
    """
    Wrapper environment for a single SAC controller.
    
    Each SAC only sees: base_state (21 dims) → params
    No strategy one-hot needed since each SAC is strategy-specific.
    """
    
    def __init__(self, base_state_dim: int, action_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(base_state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    
    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}


# =============================================================================
# Hierarchical Agent with Separate SACs
# =============================================================================

class HierarchicalAgentSAC:
    """
    Hierarchical agent with SEPARATE SAC controllers per strategy.
    
    Architecture:
    - Meta-Controller (DQN): Selects attack strategy (0=PIXEL, 1=FREQUENCY, ...)
    - Controllers (dict of SACs): Each strategy has its own SAC
    
    Benefits:
    - No interference between strategies
    - Each SAC learns optimal params for its strategy only
    - Clear separation of concerns
    """
    
    def __init__(
        self,
        env,  # HierarchicalAttackEnv
        config: Dict[str, Any] = None,
        device: str = "cuda"
    ):
        self.env = env
        self.config = config or {}
        self.device = device
        
        # Dimensions
        self.base_state_dim = env.state_extractor.STATE_DIM  # 21
        self.n_strategies = len(env.allowed_strategies)
        self.action_dim = env.action_space.shape[0]
        
        # Strategy names for logging
        self.strategy_names = [env.allowed_strategies[i].name for i in range(self.n_strategies)]
        
        # Meta-Controller (strategy selection)
        self.meta_controller = MetaController(
            state_dim=self.base_state_dim,
            n_strategies=self.n_strategies,
            config=self.config,
            device=device
        )
        
        # SEPARATE SAC controllers - one per strategy
        self.controllers: Dict[int, SAC] = {}
        self._init_sac_controllers()
        
        # Training stats
        self.total_timesteps = 0
        self.episode_count = 0
        self.strategy_usage = {name: 0 for name in self.strategy_names}
        
        # Track which controller was used (for training)
        self.current_strategy = None
    
    def _init_sac_controllers(self):
        """Initialize separate SAC controller for each strategy."""
        
        # SAC config (shared across all)
        sac_config = {
            "learning_rate": self.config.get("sac_lr", 3e-4),
            "buffer_size": self.config.get("sac_buffer_size", 50000),  # Smaller per-strategy
            "batch_size": self.config.get("sac_batch_size", 256),
            "tau": self.config.get("sac_tau", 0.005),
            "gamma": self.config.get("sac_gamma", 0.99),
            "learning_starts": self.config.get("sac_learning_starts", 500),  # Smaller
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": self.config.get("sac_ent_coef", 0.1),
            "verbose": 0,
            "device": self.device,
        }
        
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],
                "qf": [256, 256],
            }
        }
        
        print(f"\n[SAC Controllers] Initializing {self.n_strategies} separate controllers:")
        
        for strategy_idx in range(self.n_strategies):
            strategy_name = self.strategy_names[strategy_idx]
            
            # Create dummy env for this SAC (only base_state, no one-hot)
            dummy_env = SingleStrategySACEnv(
                base_state_dim=self.base_state_dim,
                action_dim=self.action_dim
            )
            
            # Create SAC for this strategy
            self.controllers[strategy_idx] = SAC(
                "MlpPolicy",
                dummy_env,
                **sac_config,
                policy_kwargs=policy_kwargs,
            )
            
            # Setup logger
            logger = configure(folder=None, format_strings=[])
            self.controllers[strategy_idx].set_logger(logger)
            
            print(f"      - SAC_{strategy_name}: obs_dim={self.base_state_dim}, action_dim={self.action_dim}")
        
        print(f"      - Buffer size per SAC: {sac_config['buffer_size']}")
        print(f"      - Total memory: ~{self.n_strategies}x single SAC")
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], int]:
        """Get action from hierarchical policy."""
        # Base state (first 21 dims)
        base_state = observation[:self.base_state_dim]
        
        # Meta-controller selects strategy
        strategy = self.meta_controller.select_strategy(base_state, deterministic)
        self.current_strategy = strategy
        
        # Use the SAC for this specific strategy
        params, _ = self.controllers[strategy].predict(base_state, deterministic=deterministic)
        
        return {"strategy": strategy, "params": params}, strategy
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[Dict[str, Any], None]:
        """Alias for get_action (SB3 compatibility)."""
        action, _ = self.get_action(observation, deterministic)
        return action, None
    
    def add_to_buffer(
        self,
        strategy: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to the correct strategy's replay buffer."""
        # Only use base state (21 dims)
        base_state = state[:self.base_state_dim]
        base_next_state = next_state[:self.base_state_dim]
        
        # Add to the specific strategy's buffer
        self.controllers[strategy].replay_buffer.add(
            base_state,
            base_next_state,
            action,
            np.array([reward]),
            np.array([done]),
            [{}]
        )
    
    def train_controller(self, strategy: int, gradient_steps: int = 1) -> Optional[float]:
        """Train the SAC for a specific strategy."""
        sac = self.controllers[strategy]
        
        if sac.replay_buffer.size() < sac.learning_starts:
            return None
        
        sac.train(gradient_steps=gradient_steps, batch_size=sac.batch_size)
        return None  # SB3 doesn't return loss easily
    
    def get_controller_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for each SAC controller."""
        stats = {}
        for strategy_idx, sac in self.controllers.items():
            strategy_name = self.strategy_names[strategy_idx]
            stats[strategy_name] = {
                "buffer_size": sac.replay_buffer.size(),
                "ent_coef": sac.ent_coef if isinstance(sac.ent_coef, float) else sac.ent_coef.item(),
            }
        return stats
    
    def set_allowed_strategies(self, strategies: List[int]):
        """Set allowed strategies for curriculum learning."""
        self.meta_controller.set_allowed_strategies(strategies)
    
    def save(self, path: str):
        """Save agent state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save meta-controller
        self.meta_controller.save(str(path / "meta_controller"))
        
        # Save each SAC controller
        for strategy_idx, sac in self.controllers.items():
            strategy_name = self.strategy_names[strategy_idx]
            sac.save(str(path / f"sac_{strategy_name.lower()}"))
        
        # Save config and stats
        stats = {
            "total_timesteps": self.total_timesteps,
            "episode_count": self.episode_count,
            "strategy_usage": self.strategy_usage,
            "base_state_dim": self.base_state_dim,
            "n_strategies": self.n_strategies,
            "action_dim": self.action_dim,
            "strategy_names": self.strategy_names,
        }
        with open(path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"[Agent] Saved to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        path = Path(path)
        
        # Load meta-controller
        self.meta_controller.load(str(path / "meta_controller"))
        
        # Load each SAC controller
        for strategy_idx in range(self.n_strategies):
            strategy_name = self.strategy_names[strategy_idx]
            sac_path = path / f"sac_{strategy_name.lower()}"
            
            if sac_path.with_suffix('.zip').exists():
                # Create dummy env for loading
                dummy_env = SingleStrategySACEnv(
                    base_state_dim=self.base_state_dim,
                    action_dim=self.action_dim
                )
                self.controllers[strategy_idx] = SAC.load(
                    str(sac_path), 
                    env=dummy_env, 
                    device=self.device
                )
                # Setup logger
                logger = configure(folder=None, format_strings=[])
                self.controllers[strategy_idx].set_logger(logger)
        
        # Load stats
        with open(path / "stats.json", "r") as f:
            stats = json.load(f)
        
        self.total_timesteps = stats["total_timesteps"]
        self.episode_count = stats["episode_count"]
        self.strategy_usage = stats["strategy_usage"]
        
        print(f"[Agent] Loaded from {path}")
        print(f"      - Timesteps: {self.total_timesteps}")
        print(f"      - Episodes: {self.episode_count}")


# =============================================================================
# Factory function
# =============================================================================

def create_hierarchical_agent_sac(
    env,
    config: Dict[str, Any] = None,
    device: str = "cuda"
) -> HierarchicalAgentSAC:
    """Create a hierarchical agent with separate SAC controllers per strategy."""
    return HierarchicalAgentSAC(env, config, device)