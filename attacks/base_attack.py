"""Base class for adversarial attacks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import numpy as np


@dataclass
class AttackResult:
    """Container for attack results."""
    
    # Original and adversarial images
    original: torch.Tensor
    adversarial: torch.Tensor
    perturbation: torch.Tensor
    
    # Labels
    original_label: torch.Tensor
    adversarial_label: torch.Tensor
    target_label: Optional[torch.Tensor] = None
    
    # Success indicators
    success: bool = False
    
    # Attack metadata
    iterations: int = 0
    queries: int = 0
    epsilon_used: float = 0.0
    strategy: str = ""
    
    # Metrics (computed later)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "iterations": self.iterations,
            "queries": self.queries,
            "epsilon_used": self.epsilon_used,
            "strategy": self.strategy,
            "original_label": self.original_label.item() if self.original_label.dim() == 0 else self.original_label.tolist(),
            "adversarial_label": self.adversarial_label.item() if self.adversarial_label.dim() == 0 else self.adversarial_label.tolist(),
            "metrics": self.metrics,
        }


class BaseAttack(ABC):
    """Abstract base class for all adversarial attacks.
    
    All attack strategies must implement this interface for
    compatibility with the RL environment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
    ):
        """Initialize attack.
        
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation magnitude (L-inf) in [0,1] range
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Dict with 'mean' and 'std' for denormalization
                            If provided, attack will work in normalized space
        """
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.targeted = targeted
        self.device = device
        
        # Normalization statistics
        self.normalize_stats = normalize_stats
        if normalize_stats is not None:
            self.mean = torch.tensor(normalize_stats['mean']).view(1, -1, 1, 1).to(device)
            self.std = torch.tensor(normalize_stats['std']).view(1, -1, 1, 1).to(device)
            # Convert epsilon to normalized space (use mean of std for simplicity)
            self.epsilon_normalized = epsilon / self.std.mean().item()
        else:
            self.mean = None
            self.std = None
            self.epsilon_normalized = epsilon
        
        # Attack statistics
        self._query_count = 0
        self._iteration_count = 0
    
    @abstractmethod
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform adversarial attack.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            target: Target labels for targeted attack [B]
            **kwargs: Attack-specific parameters
            
        Returns:
            AttackResult containing adversarial examples
        """
        pass
    
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttackResult:
        """Call attack method."""
        return self.attack(x, y, target, **kwargs)
    
    def reset_stats(self) -> None:
        """Reset query and iteration counters."""
        self._query_count = 0
        self._iteration_count = 0
    
    @property
    def query_count(self) -> int:
        """Get total number of model queries."""
        return self._query_count
    
    @property
    def iteration_count(self) -> int:
        """Get total number of attack iterations."""
        return self._iteration_count
    
    def _query_model(self, x: torch.Tensor) -> torch.Tensor:
        """Query model and track count.
        
        Args:
            x: Input images
            
        Returns:
            Model logits
        """
        self._query_count += x.shape[0]
        return self.model(x)
    
    def _query_model_no_grad(self, x):
        self._query_count += x.shape[0]
        with torch.no_grad():
            return self.model(x)

    
    def _get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get model predictions.
        
        Args:
            x: Input images
            
        Returns:
            Predicted labels
        """
        logits = self._query_model_no_grad(x)
        return logits.argmax(dim=1)
    
    def _check_success(
        self,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Check if attack was successful.
        
        Args:
            x_adv: Adversarial images
            y: True labels
            target: Target labels (for targeted attack)
            
        Returns:
            Boolean tensor indicating success per sample
        """
        preds = self._get_predictions(x_adv)
        
        if self.targeted and target is not None:
            # Success if prediction equals target
            return preds == target
        else:
            # Success if prediction differs from true label
            return preds != y
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize images to [0,1] range.
        
        Args:
            x: Normalized images
            
        Returns:
            Denormalized images in [0,1]
        """
        if self.mean is not None and self.std is not None:
            return x * self.std + self.mean
        return x
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images from [0,1] range.
        
        Args:
            x: Images in [0,1] range
            
        Returns:
            Normalized images
        """
        if self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        return x
    
    def _project_perturbation(
        self,
        perturbation: torch.Tensor,
        epsilon: Optional[float] = None,
        norm: str = "linf",
    ) -> torch.Tensor:
        """Project perturbation onto epsilon ball.
        
        If normalize_stats is provided, projects in [0,1] space to ensure
        epsilon budget is respected in pixel space.
        
        Args:
            perturbation: Perturbation tensor
            epsilon: Perturbation budget (default: self.epsilon)
            norm: Norm type ('linf', 'l2', 'l1')
            
        Returns:
            Projected perturbation
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # If normalized, use denormalized projection
        if self.normalize_stats is not None:
            return self._project_perturbation_denormalized(perturbation, epsilon, norm)
        
        if norm == "linf":
            return torch.clamp(perturbation, -epsilon, epsilon)
        elif norm == "l2":
            # Project onto L2 ball
            norms = perturbation.view(perturbation.shape[0], -1).norm(dim=1, keepdim=True)
            norms = norms.view(-1, 1, 1, 1)
            factor = torch.min(torch.ones_like(norms), epsilon / (norms + 1e-12))
            return perturbation * factor
        elif norm == "l1":
            # Project onto L1 ball (more complex)
            raise NotImplementedError("L1 projection not yet implemented")
        else:
            raise ValueError(f"Unknown norm: {norm}")
    
    def _project_perturbation_denormalized(
        self,
        perturbation: torch.Tensor,
        epsilon: Optional[float] = None,
        norm: str = "linf",
    ) -> torch.Tensor:
        """Project perturbation onto epsilon ball in denormalized space.
        
        This ensures perturbations respect epsilon in [0,1] space even when
        working in normalized space.
        
        Args:
            perturbation: Perturbation tensor in normalized space
            epsilon: Perturbation budget in [0,1] space
            norm: Norm type ('linf', 'l2')
            
        Returns:
            Projected perturbation in normalized space
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Denormalize perturbation to [0,1] space
        if self.std is not None:
            pert_denorm = perturbation * self.std
        else:
            pert_denorm = perturbation
        
        # Project in [0,1] space
        if norm == "linf":
            pert_denorm = torch.clamp(pert_denorm, -epsilon, epsilon)
        elif norm == "l2":
            norms = pert_denorm.view(pert_denorm.shape[0], -1).norm(dim=1, keepdim=True)
            norms = norms.view(-1, 1, 1, 1)
            factor = torch.min(torch.ones_like(norms), epsilon / (norms + 1e-12))
            pert_denorm = pert_denorm * factor
        else:
            raise ValueError(f"Unknown norm: {norm}")
        
        # Normalize back
        if self.std is not None:
            return pert_denorm / self.std
        return pert_denorm
    
    def _clamp_image(
        self,
        x: torch.Tensor,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> torch.Tensor:
        """Clamp image to valid range, handling normalization.
        
        If normalize_stats is provided, clamps in [0,1] space then re-normalizes.
        Otherwise clamps directly.
        
        Args:
            x: Image tensor (may be normalized)
            min_val: Minimum value in [0,1] space
            max_val: Maximum value in [0,1] space
            
        Returns:
            Clamped image (in same space as input)
        """
        if self.normalize_stats is not None:
            # Denormalize to [0,1], clamp, re-normalize
            x_denorm = self._denormalize(x)
            x_denorm = torch.clamp(x_denorm, min_val, max_val)
            return self._normalize(x_denorm)
        else:
            return torch.clamp(x, min_val, max_val)
    
    def _compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient of loss w.r.t. input.
        
        Args:
            x: Input images (will be modified to require grad)
            y: True labels
            target: Target labels (for targeted attack)
            
        Returns:
            Gradient tensor
        """
        x_var = x.clone().detach().requires_grad_(True)
        
        #logits = self.model(x_var)
        logits = self._query_model(x_var)


        if self.targeted and target is not None:
            # Minimize loss for target class (gradient descent)
            loss = nn.functional.cross_entropy(logits, target)
        else:
            # Maximize loss for true class (gradient ascent)
            # Use positive CE - the sign is handled in the attack methods
            loss = nn.functional.cross_entropy(logits, y)
        
        loss.backward()
        grad = x_var.grad.detach()
        
        #self._query_count += x.shape[0]
        
        return grad
    
    def set_epsilon(self, epsilon: float) -> None:
        """Update epsilon value.
        
        Args:
            epsilon: New epsilon value
        """
        self.epsilon = epsilon
        if self.std is not None:
            self.epsilon_normalized = epsilon / self.std.mean().item()
        else:
            self.epsilon_normalized = epsilon
    
    def set_targeted(self, targeted: bool) -> None:
        """Set targeted/untargeted mode.
        
        Args:
            targeted: Whether to use targeted attack
        """
        self.targeted = targeted
    
    @staticmethod
    def perturbation_norm(perturbation: torch.Tensor, norm: str = "linf") -> torch.Tensor:
        """Compute perturbation norm.
        
        Args:
            perturbation: Perturbation tensor
            norm: Norm type
            
        Returns:
            Norm value per sample
        """
        flat = perturbation.view(perturbation.shape[0], -1)
        
        if norm == "linf":
            return flat.abs().max(dim=1)[0]
        elif norm == "l2":
            return flat.norm(dim=1)
        elif norm == "l1":
            return flat.abs().sum(dim=1)
        elif norm == "l0":
            return (flat.abs() > 1e-8).float().sum(dim=1)
        else:
            raise ValueError(f"Unknown norm: {norm}")