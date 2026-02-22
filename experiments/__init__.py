"""
Experiments module for MedSecure.

Contains scripts for:
- Training victim models on MedMNIST datasets
- Running baseline attacks
- Training the RL agent
- Comprehensive evaluation
- Ablation studies
"""

from .train_victim import train_victim_model, VictimTrainer
from .run_baselines import run_baseline_attacks, BaselineRunner
from .train_agent import train_agent, AgentTrainer
from .evaluate import run_evaluation, Evaluator
from .ablation import run_ablation_study, AblationRunner

__all__ = [
    "train_victim_model",
    "VictimTrainer",
    "run_baseline_attacks",
    "BaselineRunner",
    "train_agent",
    "AgentTrainer",
    "run_evaluation",
    "Evaluator",
    "run_ablation_study",
    "AblationRunner"
]
