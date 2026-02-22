"""
Curriculum Learning Scheduler for Adversarial Attack Training.

This module implements curriculum learning strategies that progressively
increase task difficulty during training:

Phase 0: Easy - Large epsilon, simple attacks
Phase 1: Medium - Reduced epsilon, multiple strategies
Phase 2: Hard - Small epsilon, full strategy space
Phase 3: Expert - Minimal epsilon, emphasis on imperceptibility

The scheduler monitors training metrics and adjusts difficulty accordingly.
"""

import json
import sys
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import IntEnum


def _log(msg: str):
    """Print with immediate flush for visibility."""
    print(msg, flush=True)


class CurriculumPhase(IntEnum):
    """Curriculum learning phases."""
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXPERT = 3


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""
    name: str
    epsilon_range: tuple  # (min, max)
    allowed_strategies: List[str]
    max_iterations: int
    success_threshold: float  # Required success rate to advance
    min_episodes: int  # Minimum episodes before advancement
    reward_weights: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "epsilon_range": self.epsilon_range,
            "allowed_strategies": self.allowed_strategies,
            "max_iterations": self.max_iterations,
            "success_threshold": self.success_threshold,
            "min_episodes": self.min_episodes,
            "reward_weights": self.reward_weights
        }


# Default phase configurations
DEFAULT_PHASES = [
    PhaseConfig(
        name="easy",
        epsilon_range=(0.1, 0.3),
        allowed_strategies=["pixel"],
        max_iterations=50,
        success_threshold=0.6,
        min_episodes=5,  # Called every 1000 timesteps, so 5 = 5000 timesteps
        reward_weights={
            "success": 1.0,
            "imperceptibility": 0.1,
            "efficiency": 0.1,
            "diversity": 0.0  # Only PIXEL available, no diversity needed
        }
    ),
    PhaseConfig(
        name="medium",
        epsilon_range=(0.05, 0.2),
        allowed_strategies=["pixel", "frequency"],
        max_iterations=75,
        success_threshold=0.5,
        min_episodes=10,  # 10000 timesteps
        reward_weights={
            "success": 1.0,
            "imperceptibility": 0.2,
            "efficiency": 0.15,
            "diversity": 0.2  # Increased to encourage FREQUENCY exploration
        }
    ),
    PhaseConfig(
        name="hard",
        epsilon_range=(0.02, 0.1),
        allowed_strategies=["pixel", "frequency", "semantic"],
        max_iterations=100,
        success_threshold=0.4,
        min_episodes=15,  # 15000 timesteps
        reward_weights={
            "success": 1.0,
            "imperceptibility": 0.3,
            "efficiency": 0.2,
            "diversity": 0.25  # Increased to encourage SEMANTIC exploration
        }
    ),
    PhaseConfig(
        name="expert",
        epsilon_range=(0.01, 0.05),
        allowed_strategies=["pixel", "frequency", "semantic"],
        max_iterations=100,
        success_threshold=0.3,
        min_episodes=20,  # 20000 timesteps
        reward_weights={
            "success": 1.0,
            "imperceptibility": 0.4,
            "efficiency": 0.2,
            "diversity": 0.2  # Keep diversity important
        }
    )
]


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive training difficulty.
    
    The scheduler tracks training progress and automatically advances
    through phases based on performance metrics. It can also be manually
    controlled for ablation studies.
    """
    
    def __init__(
        self,
        phases: Optional[List[PhaseConfig]] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_advance: bool = True,
        log_dir: Optional[str] = None
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            phases: List of phase configurations (uses defaults if None)
            config: Configuration dictionary to extract phases from
            auto_advance: Automatically advance phases based on metrics
            log_dir: Directory for logging phase transitions
        """
        # Extract phases from config or use defaults
        if phases is not None:
            self.phases = phases
        elif config is not None:
            self.phases = self._parse_phases_from_config(config)
        else:
            self.phases = DEFAULT_PHASES
        
        self.auto_advance = auto_advance
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Create log directory if specified
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_phase = 0
        self.phase_episodes = 0
        self.phase_successes = []
        self.total_episodes = 0
        
        # History
        self.phase_history = []
        self.transition_log = []
        
        # Callbacks
        self.on_phase_change_callbacks: List[Callable] = []
    
    def _parse_phases_from_config(
        self,
        config: Dict[str, Any]
    ) -> List[PhaseConfig]:
        """Parse phase configurations from config dictionary."""
        curriculum_config = config.get("rl", {}).get("curriculum", {})
        phases_config = curriculum_config.get("phases", [])
        
        if not phases_config:
            return DEFAULT_PHASES
        
        phases = []
        for pc in phases_config:
            phases.append(PhaseConfig(
                name=pc.get("name", f"phase_{len(phases)}"),
                epsilon_range=tuple(pc.get("epsilon_range", (0.1, 0.3))),
                allowed_strategies=pc.get("allowed_strategies", ["pixel"]),
                max_iterations=pc.get("max_iterations", 50),
                success_threshold=pc.get("success_threshold", 0.5),
                min_episodes=pc.get("min_episodes", 1000),
                reward_weights=pc.get("reward_weights", {})
            ))
        
        return phases
    
    @property
    def phase(self) -> CurriculumPhase:
        """Get current curriculum phase as enum."""
        return CurriculumPhase(min(self.current_phase, len(CurriculumPhase) - 1))
    
    @property
    def phase_config(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self.phases[self.current_phase]
    
    @property
    def is_final_phase(self) -> bool:
        """Check if we're in the final phase."""
        return self.current_phase >= len(self.phases) - 1
    
    def get_epsilon_range(self) -> tuple:
        """Get current epsilon range."""
        return self.phase_config.epsilon_range
    
    def get_allowed_strategies(self) -> List[str]:
        """Get currently allowed attack strategies."""
        return self.phase_config.allowed_strategies
    
    def get_max_iterations(self) -> int:
        """Get maximum iterations for current phase."""
        return self.phase_config.max_iterations
    
    def get_reward_weights(self) -> Dict[str, float]:
        """Get reward weights for current phase."""
        return self.phase_config.reward_weights
    
    def step(self, metrics: Dict[str, float]) -> bool:
        """
        Update scheduler with new training metrics.
        
        Args:
            metrics: Dictionary containing 'success_rate' and optionally other metrics
            
        Returns:
            True if phase changed, False otherwise
        """
        self.total_episodes += 1
        self.phase_episodes += 1
        
        # Track success
        success_rate = metrics.get("success_rate", 0.0)
        self.phase_successes.append(success_rate)
        
        # Record history
        self.phase_history.append({
            "episode": self.total_episodes,
            "phase": self.current_phase,
            "success_rate": success_rate,
            "metrics": metrics
        })
        
        # Check for phase advancement
        if self.auto_advance and self._should_advance():
            self._advance_phase()
            return True
        
        return False
    
    def _should_advance(self) -> bool:
        """Check if we should advance to next phase."""
        if self.is_final_phase:
            return False
        
        config = self.phase_config
        
        # Need minimum episodes (evaluations)
        if self.phase_episodes < config.min_episodes:
            return False
        
        # Need at least some success data
        if len(self.phase_successes) < 5:
            return False
        
        # Use last 100 or all available if less
        n_samples = min(100, len(self.phase_successes))
        recent_success_rate = np.mean(self.phase_successes[-n_samples:])
        
        # Log progress periodically
        if self.phase_episodes % 5 == 0:
            _log(f"  [Curriculum] Phase {self.current_phase} ({config.name}): "
                  f"episodes={self.phase_episodes}/{config.min_episodes}, "
                  f"success_rate={recent_success_rate:.1%} (need {config.success_threshold:.0%})")
        
        return recent_success_rate >= config.success_threshold
    
    def _advance_phase(self):
        """Advance to next curriculum phase."""
        old_phase = self.current_phase
        self.current_phase = min(self.current_phase + 1, len(self.phases) - 1)
        
        # Log transition
        transition = {
            "episode": self.total_episodes,
            "from_phase": old_phase,
            "to_phase": self.current_phase,
            "episodes_in_phase": self.phase_episodes,
            "final_success_rate": np.mean(self.phase_successes[-100:]) if self.phase_successes else 0
        }
        self.transition_log.append(transition)
        
        _log(f"\n{'='*50}")
        _log(f"CURRICULUM PHASE TRANSITION: {self.phases[old_phase].name} -> {self.phase_config.name}")
        _log(f"Episodes in previous phase: {self.phase_episodes}")
        _log(f"Success rate: {transition['final_success_rate']:.2%}")
        _log(f"New epsilon range: {self.phase_config.epsilon_range}")
        _log(f"New strategies: {self.phase_config.allowed_strategies}")
        _log(f"{'='*50}\n")
        
        # Save transition to file
        if self.log_dir:
            log_file = Path(self.log_dir) / "curriculum_transitions.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(transition) + "\n")
        
        # Reset phase counters
        self.phase_episodes = 0
        self.phase_successes = []
        
        # Call callbacks
        for callback in self.on_phase_change_callbacks:
            callback(old_phase, self.current_phase)
        
        # Save log
        if self.log_dir:
            self._save_transition_log()
    
    def force_advance(self) -> bool:
        """Force advancement to next phase."""
        if self.is_final_phase:
            return False
        
        self._advance_phase()
        return True
    
    def set_phase(self, phase: int):
        """Manually set curriculum phase."""
        if 0 <= phase < len(self.phases):
            old_phase = self.current_phase
            self.current_phase = phase
            self.phase_episodes = 0
            self.phase_successes = []
            
            print(f"Manually set phase: {self.phases[old_phase].name} -> {self.phase_config.name}")
    
    def register_callback(self, callback: Callable[[int, int], None]):
        """Register callback for phase changes."""
        self.on_phase_change_callbacks.append(callback)
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "current_phase": self.current_phase,
            "phase_episodes": self.phase_episodes,
            "total_episodes": self.total_episodes,
            "phase_successes": self.phase_successes[-100:],  # Keep recent
            "transition_log": self.transition_log
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        self.current_phase = state.get("current_phase", 0)
        self.phase_episodes = state.get("phase_episodes", 0)
        self.total_episodes = state.get("total_episodes", 0)
        self.phase_successes = state.get("phase_successes", [])
        self.transition_log = state.get("transition_log", [])
    
    def _save_transition_log(self):
        """Save transition log to disk."""
        if self.log_dir is None:
            return
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "curriculum_transitions.json"
        
        with open(log_path, "w") as f:
            json.dump({
                "transitions": self.transition_log,
                "phases": [p.to_dict() for p in self.phases],
                "current_phase": self.current_phase
            }, f, indent=2)
    
    def get_progress_string(self) -> str:
        """Get human-readable progress string."""
        recent_success = np.mean(self.phase_successes[-100:]) if len(self.phase_successes) >= 100 else 0
        
        return (
            f"Phase: {self.phase_config.name} ({self.current_phase + 1}/{len(self.phases)}) | "
            f"Episodes: {self.phase_episodes}/{self.phase_config.min_episodes} | "
            f"Success: {recent_success:.1%}/{self.phase_config.success_threshold:.1%}"
        )


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum scheduler that can also regress to easier phases.
    
    If performance drops significantly, the scheduler can move back to
    an easier phase to stabilize learning.
    """
    
    def __init__(
        self,
        regression_threshold: float = 0.2,
        regression_lookback: int = 500,
        **kwargs
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            regression_threshold: Performance drop that triggers regression
            regression_lookback: Episodes to look back for regression check
            **kwargs: Arguments for base CurriculumScheduler
        """
        super().__init__(**kwargs)
        
        self.regression_threshold = regression_threshold
        self.regression_lookback = regression_lookback
        self.peak_performance = 0.0
        self.regression_count = 0
        self.max_regressions = 3  # Limit regressions to avoid oscillation
    
    def step(self, metrics: Dict[str, float]) -> bool:
        """Update with regression checking."""
        # Normal step
        phase_changed = super().step(metrics)
        
        # Check for regression
        if not phase_changed and self._should_regress():
            self._regress_phase()
            return True
        
        # Update peak performance
        if len(self.phase_successes) >= 100:
            current_rate = np.mean(self.phase_successes[-100:])
            self.peak_performance = max(self.peak_performance, current_rate)
        
        return phase_changed
    
    def _should_regress(self) -> bool:
        """Check if we should regress to easier phase."""
        if self.current_phase == 0:
            return False
        
        if self.regression_count >= self.max_regressions:
            return False
        
        if len(self.phase_successes) < self.regression_lookback:
            return False
        
        recent_rate = np.mean(self.phase_successes[-100:])
        older_rate = np.mean(self.phase_successes[-self.regression_lookback:-100])
        
        # Check for significant performance drop
        performance_drop = older_rate - recent_rate
        
        return performance_drop > self.regression_threshold
    
    def _regress_phase(self):
        """Regress to previous phase."""
        old_phase = self.current_phase
        self.current_phase = max(0, self.current_phase - 1)
        self.regression_count += 1
        
        print(f"\n{'!'*50}")
        print(f"CURRICULUM REGRESSION: {self.phases[old_phase].name} -> {self.phase_config.name}")
        print(f"Regression #{self.regression_count}/{self.max_regressions}")
        print(f"{'!'*50}\n")
        
        # Reset counters
        self.phase_episodes = 0
        self.phase_successes = []
        self.peak_performance = 0.0
        
        # Log transition
        self.transition_log.append({
            "episode": self.total_episodes,
            "from_phase": old_phase,
            "to_phase": self.current_phase,
            "type": "regression"
        })


def create_scheduler(
    config: Dict[str, Any],
    adaptive: bool = False,
    log_dir: Optional[str] = None
) -> CurriculumScheduler:
    """
    Factory function to create curriculum scheduler.
    
    Args:
        config: Configuration dictionary
        adaptive: Use adaptive scheduler with regression
        log_dir: Log directory
        
    Returns:
        Configured CurriculumScheduler
    """
    scheduler_class = AdaptiveCurriculumScheduler if adaptive else CurriculumScheduler
    
    return scheduler_class(
        config=config,
        auto_advance=config.get("rl", {}).get("curriculum", {}).get("auto_advance", True),
        log_dir=log_dir
    )