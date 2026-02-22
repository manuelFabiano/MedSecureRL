"""Fast Gradient Sign Method (FGSM) baseline attack."""

from typing import Optional
import torch
import torch.nn as nn

from ..base_attack import BaseAttack, AttackResult


class FGSM(BaseAttack):
    """Fast Gradient Sign Method.
    
    Single-step gradient-based attack.
    
    Reference:
        Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[dict] = None,
    ):
        """Initialize FGSM.
        
        Args:
            model: Target model
            epsilon: Perturbation magnitude in [0,1] range
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Normalization statistics (mean, std)
        """
        super().__init__(model, epsilon, targeted, device, normalize_stats)
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform FGSM attack.
        
        Args:
            x: Input images [B, C, H, W] (possibly normalized)
            y: True labels [B]
            target: Target labels for targeted attack
            epsilon: Override epsilon (in [0,1] range)
            
        Returns:
            AttackResult
        """
        eps = epsilon if epsilon is not None else self.epsilon
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # Compute gradient
        grad = self._compute_gradient(x, y, target)
        
        # Create perturbation in normalized space (if applicable)
        # For untargeted: we want gradient ASCENT on CE, so we ADD gradient
        # For targeted: we want gradient DESCENT on CE(target), so we SUBTRACT gradient
        if self.targeted:
            perturbation = -self.epsilon_normalized * grad.sign()
        else:
            perturbation = self.epsilon_normalized * grad.sign()
        
        # Project perturbation to respect epsilon in [0,1] space
        perturbation = self._project_perturbation_denormalized(perturbation, eps)
        
        # Create adversarial example
        x_adv = x + perturbation
        
        # Clamp in [0,1] space (denormalize, clamp, renormalize)
        if self.normalize_stats is not None:
            x_adv_denorm = self._denormalize(x_adv)
            x_adv_denorm = torch.clamp(x_adv_denorm, 0.0, 1.0)
            x_adv = self._normalize(x_adv_denorm)
        else:
            x_adv = self._clamp_image(x_adv, 0.0, 1.0)
        
        # Recompute actual perturbation after clamping
        perturbation = x_adv - x
        
        # Check success
        adv_preds = self._get_predictions(x_adv)
        success = self._check_success(x_adv, y, target)
        
        return AttackResult(
            original=x,
            adversarial=x_adv,
            perturbation=perturbation,
            original_label=y,
            adversarial_label=adv_preds,
            target_label=target,
            success=success.all().item(),
            iterations=1,
            queries=self._query_count,
            epsilon_used=eps,
            strategy="fgsm",
        )