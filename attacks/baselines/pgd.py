"""Projected Gradient Descent (PGD) and BIM baseline attacks."""

from typing import Optional
import torch
import torch.nn as nn

from ..base_attack import BaseAttack, AttackResult


class PGD(BaseAttack):
    """Projected Gradient Descent attack.
    
    Multi-step attack with random initialization.
    
    Reference:
        Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        iterations: int = 40,
        step_size: Optional[float] = None,
        random_start: bool = True,
        restarts: int = 1,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[dict] = None,
    ):
        """Initialize PGD.
        
        Args:
            model: Target model
            epsilon: Maximum perturbation magnitude in [0,1] range
            iterations: Number of iterations
            step_size: Step size per iteration (default: epsilon/4)
            random_start: Use random initialization
            restarts: Number of random restarts
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Normalization statistics (mean, std)
        """
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        
        self.iterations = iterations
        self.step_size = step_size if step_size is not None else epsilon / 4
        self.random_start = random_start
        self.restarts = restarts
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        iterations: Optional[int] = None,
        step_size: Optional[float] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform PGD attack.
        
        Args:
            x: Input images [B, C, H, W] (possibly normalized)
            y: True labels [B]
            target: Target labels for targeted attack
            epsilon: Override epsilon (in [0,1] range)
            iterations: Override iterations
            step_size: Override step size
            
        Returns:
            AttackResult
        """
        eps = epsilon if epsilon is not None else self.epsilon
        iters = iterations if iterations is not None else self.iterations
        alpha = step_size if step_size is not None else self.step_size
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        best_x_adv = None
        best_loss = float('-inf')
        
        for restart in range(self.restarts):
            x_adv = self._pgd_attack(x, y, target, eps, iters, alpha)
            
            # Evaluate this restart
            with torch.no_grad():
                logits = self._query_model(x_adv)
                if self.targeted and target is not None:
                    loss = -nn.functional.cross_entropy(logits, target)
                else:
                    loss = nn.functional.cross_entropy(logits, y)
            
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_x_adv = x_adv
        
        # Compute perturbation
        perturbation = best_x_adv - x
        
        # Check success
        adv_preds = self._get_predictions(best_x_adv)
        success = self._check_success(best_x_adv, y, target)
        
        return AttackResult(
            original=x,
            adversarial=best_x_adv,
            perturbation=perturbation,
            original_label=y,
            adversarial_label=adv_preds,
            target_label=target,
            success=success.all().item(),
            iterations=iters * self.restarts,
            queries=self._query_count,
            epsilon_used=eps,
            strategy="pgd",
        )
    
    def _pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        epsilon: float,
        iterations: int,
        step_size: float,
    ) -> torch.Tensor:
        """Single PGD attack run.
        
        Args:
            x: Input images (possibly normalized)
            y: True labels
            target: Target labels
            epsilon: Perturbation budget in [0,1] range
            iterations: Number of iterations
            step_size: Step size in [0,1] range
            
        Returns:
            Adversarial images
        """
        x_adv = x.clone()
        
        # Random initialization in normalized space
        if self.random_start:
            # Convert epsilon to normalized space for initialization
            eps_norm = self.epsilon_normalized
            noise = torch.empty_like(x).uniform_(-eps_norm, eps_norm)
            x_adv = x_adv + noise
            
            # Clamp in [0,1] space
            if self.normalize_stats is not None:
                x_adv_denorm = self._denormalize(x_adv)
                x_adv_denorm = torch.clamp(x_adv_denorm, 0.0, 1.0)
                x_adv = self._normalize(x_adv_denorm)
            else:
                x_adv = self._clamp_image(x_adv, 0.0, 1.0)
        
        # Convert step size to normalized space
        if self.normalize_stats is not None:
            alpha_normalized = step_size / self.std.mean().item()
        else:
            alpha_normalized = step_size
        
        for i in range(iterations):
            self._iteration_count = i + 1
            
            # Compute gradient
            grad = self._compute_gradient(x_adv, y, target)
            
            # Update in normalized space
            # For untargeted: gradient ASCENT on CE, so we ADD gradient
            # For targeted: gradient DESCENT on CE(target), so we SUBTRACT gradient
            if self.targeted:
                x_adv = x_adv - alpha_normalized * grad.sign()
            else:
                x_adv = x_adv + alpha_normalized * grad.sign()
            
            # Project perturbation to respect epsilon in [0,1] space
            perturbation = x_adv - x
            perturbation = self._project_perturbation_denormalized(perturbation, epsilon)
            x_adv = x + perturbation
            
            # Clamp in [0,1] space
            if self.normalize_stats is not None:
                x_adv_denorm = self._denormalize(x_adv)
                x_adv_denorm = torch.clamp(x_adv_denorm, 0.0, 1.0)
                x_adv = self._normalize(x_adv_denorm)
            else:
                x_adv = self._clamp_image(x_adv, 0.0, 1.0)
        
        return x_adv


class BIM(PGD):
    """Basic Iterative Method.
    
    Like PGD but without random initialization.
    
    Reference:
        Kurakin et al., "Adversarial examples in the physical world", 2017
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        iterations: int = 40,
        step_size: Optional[float] = None,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[dict] = None,
    ):
        """Initialize BIM.
        
        Args:
            model: Target model
            epsilon: Maximum perturbation magnitude in [0,1] range
            iterations: Number of iterations
            step_size: Step size per iteration
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Normalization statistics (mean, std)
        """
        super().__init__(
            model=model,
            epsilon=epsilon,
            iterations=iterations,
            step_size=step_size,
            random_start=False,  # Main difference from PGD
            restarts=1,
            targeted=targeted,
            device=device,
            normalize_stats=normalize_stats,
        )
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform BIM attack."""
        result = super().attack(x, y, target, **kwargs)
        result.strategy = "bim"
        return result