"""Victim model definitions for MedSecure."""

from .victim_models import (
    get_victim_model,
    VictimModel,
    ResNetVictim,
    DenseNetVictim,
    EfficientNetVictim,
    SUPPORTED_MODELS,
)

__all__ = [
    "get_victim_model",
    "VictimModel",
    "ResNetVictim",
    "DenseNetVictim",
    "EfficientNetVictim",
    "SUPPORTED_MODELS",
]
