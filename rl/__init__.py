"""
Reinforcement Learning components for MedSecure.

This module provides the RL infrastructure for training an adversarial
attack generation agent:
- Environment: Gymnasium-compatible MDP for attack selection
- State Extractor: Feature extraction from images and models
- Reward: Multi-objective reward function
- Agent: SAC-based agent with curriculum learning
- Curriculum: Progressive difficulty scheduling
"""

from .state_extractor import StateExtractor
from .environment import AdversarialAttackEnv, AttackStrategy, AttackAction, make_env
from .reward import RewardFunction, RewardComponents, MedicalImperceptibilityReward
from .agent import MedSecureAgent, AttackFeaturesExtractor, create_agent
from .curriculum import (
    CurriculumScheduler,
    AdaptiveCurriculumScheduler,
    CurriculumPhase,
    PhaseConfig,
    create_scheduler
)

__all__ = [
    # State
    "StateExtractor",
    # Environment
    "AdversarialAttackEnv",
    "AttackStrategy",
    "AttackAction",
    "make_env",
    # Reward
    "RewardFunction",
    "RewardComponents",
    "MedicalImperceptibilityReward",
    # Agent
    "MedSecureAgent",
    "AttackFeaturesExtractor",
    "create_agent",
    # Curriculum
    "CurriculumScheduler",
    "AdaptiveCurriculumScheduler",
    "CurriculumPhase",
    "PhaseConfig",
    "create_scheduler"
]
