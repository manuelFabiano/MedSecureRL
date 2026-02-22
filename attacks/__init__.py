"""Adversarial attack implementations for MedSecure."""

from .base_attack import BaseAttack, AttackResult
from .pixel_attack import PixelAttack
from .frequency_attack import FrequencyAttack
from .patch_attack import PatchAttack
from .semantic_attack import SemanticAttack
from .square_attack import SquareAttack

from .baselines import (
    FGSM,
    PGD,
    BIM,
    CarliniWagner,
    AutoAttackWrapper,
)

__all__ = [
    "BaseAttack", "AttackResult",
    "PixelAttack", "FrequencyAttack", "PatchAttack", "SemanticAttack", "SquareAttack",
    "FGSM", "PGD", "BIM", "CarliniWagner", "AutoAttackWrapper",
]


ATTACK_REGISTRY = {
    "pixel": PixelAttack,
    "frequency": FrequencyAttack,
    "patch": PatchAttack,
    "semantic": SemanticAttack,
    "square": SquareAttack,
}


def get_attack(name: str, **kwargs) -> BaseAttack:
    """Get attack by name.
    
    Args:
        name: Attack name
        **kwargs: Attack-specific arguments
        
    Returns:
        Attack instance
    """
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[name](**kwargs)