"""Square Attack: query-based black-box L-inf attack.

This is a score-based attack that does NOT use gradients.
It works by randomly modifying square-shaped regions and keeping
changes that improve the attack objective.

Effective against adversarially trained models where gradient-based
attacks fail due to gradient masking/obfuscation.

Reference:
    Andriushchenko et al., "Square Attack: a query-efficient black-box
    adversarial attack via random search", ECCV 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List

from .base_attack import BaseAttack, AttackResult


class SquareAttack(BaseAttack):
    """Score-based black-box L-inf attack using random square perturbations.
    
    The RL agent controls:
    - epsilon: perturbation magnitude
    - n_queries: maximum number of queries (budget)
    - p_init: initial fraction of pixels to perturb per step
    
    No gradients are used. The attack only needs forward passes (scores).
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        n_queries: int = 1000,
        p_init: float = 0.8,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        self.n_queries = n_queries
        self.p_init = p_init
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        n_queries: Optional[int] = None,
        p_init: Optional[float] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform Square Attack.
        
        Args:
            x: Input images [B, C, H, W] (possibly normalized)
            y: True labels [B]
            target: Not used (untargeted only)
            epsilon: L-inf budget in pixel space
            n_queries: Max number of model queries
            p_init: Initial fraction of pixels to modify
            
        Returns:
            AttackResult
        """
        eps = epsilon if epsilon is not None else self.epsilon
        queries = n_queries if n_queries is not None else self.n_queries
        p0 = p_init if p_init is not None else self.p_init
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Work in pixel space [0,1] for clean epsilon enforcement
        if self.normalize_stats is not None:
            x_pixel = self._denormalize(x)
        else:
            x_pixel = x.clone()
        
        x_adv_pixel = self._square_attack_linf(x_pixel, y, eps, queries, p0)
        
        # Convert back to normalized space
        if self.normalize_stats is not None:
            x_adv = self._normalize(x_adv_pixel)
        else:
            x_adv = x_adv_pixel
        
        # Results
        perturbation = x_adv - x
        adv_preds = self._get_predictions(x_adv)
        success = self._check_success(x_adv, y, None)
        
        return AttackResult(
            original=x,
            adversarial=x_adv,
            perturbation=perturbation,
            original_label=y,
            adversarial_label=adv_preds,
            target_label=None,
            success=success.all().item(),
            iterations=self._iteration_count,
            queries=self._query_count,
            epsilon_used=eps,
            strategy="square",
        )
    
    def _margin_loss(self, x_norm: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute margin loss: f(y) - max_{c != y} f(c).
        
        Negative margin = successful attack.
        
        Args:
            x_norm: Images in normalized space (for model input)
            y: True labels
            
        Returns:
            Margin loss per sample [B]
        """
        logits = self._query_model_no_grad(x_norm)
        
        # Get logit of true class
        true_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        
        # Get max logit of non-true classes
        logits_masked = logits.clone()
        logits_masked.scatter_(1, y.unsqueeze(1), float('-inf'))
        max_other = logits_masked.max(dim=1)[0]
        
        # Margin: positive means correctly classified, negative means misclassified
        return true_logits - max_other
    
    def _p_schedule(self, step: int, n_queries: int, p_init: float) -> float:
        """Compute fraction of pixels to modify at current step.
        
        Follows the schedule from the original paper:
        starts large (p_init) and decreases over time.
        """
        # Piecewise constant schedule from original paper
        periods = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        reductions = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        progress = step / max(n_queries, 1)
        
        for i in range(len(periods) - 1):
            if progress >= periods[i] and progress < periods[i + 1]:
                return p_init * reductions[i]
        
        return p_init * reductions[-1]
    
    def _square_attack_linf(
        self,
        x_pixel: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        n_queries: int,
        p_init: float,
    ) -> torch.Tensor:
        """Core Square Attack in pixel space [0,1].
        
        Args:
            x_pixel: Original images in [0,1]
            y: True labels
            epsilon: L-inf budget
            n_queries: Query budget
            p_init: Initial pixel modification fraction
            
        Returns:
            Adversarial images in [0,1]
        """
        batch_size, c, h, w = x_pixel.shape
        
        # Initialize with random perturbation at epsilon corners {-eps, +eps}
        init_pert = torch.sign(torch.randn(batch_size, c, 1, w, device=self.device))
        init_pert = init_pert.expand(-1, -1, h, -1) * epsilon
        
        x_adv = torch.clamp(x_pixel + init_pert, 0.0, 1.0)
        
        # Compute initial margin
        if self.normalize_stats is not None:
            x_adv_norm = self._normalize(x_adv)
        else:
            x_adv_norm = x_adv
        
        best_margin = self._margin_loss(x_adv_norm, y)
        best_x_adv = x_adv.clone()
        
        # Track which samples are already successful
        successful = (best_margin < 0)
        
        for step in range(n_queries):
            self._iteration_count = step + 1
            
            # All done?
            if successful.all():
                break
            
            # Current p (fraction of image to modify)
            p = self._p_schedule(step, n_queries, p_init)
            
            # Square side length
            s = max(1, int(round(np.sqrt(p * h * w))))
            s = min(s, h, w)
            
            # Random position for the square
            top = np.random.randint(0, max(1, h - s + 1))
            left = np.random.randint(0, max(1, w - s + 1))
            
            # Random values for the square patch: {-eps, +eps}
            patch_values = torch.sign(torch.randn(batch_size, c, s, s, device=self.device)) * epsilon
            
            # Create candidate: modify the square region
            x_candidate = best_x_adv.clone()
            
            # Apply: set the square to original + patch
            x_candidate[:, :, top:top+s, left:left+s] = torch.clamp(
                x_pixel[:, :, top:top+s, left:left+s] + patch_values,
                0.0, 1.0
            )
            
            # Ensure overall L-inf constraint
            delta = x_candidate - x_pixel
            delta = torch.clamp(delta, -epsilon, epsilon)
            x_candidate = torch.clamp(x_pixel + delta, 0.0, 1.0)
            
            # Evaluate candidate
            if self.normalize_stats is not None:
                x_cand_norm = self._normalize(x_candidate)
            else:
                x_cand_norm = x_candidate
            
            cand_margin = self._margin_loss(x_cand_norm, y)
            
            # Keep if margin improved (lower is better for attack)
            improved = cand_margin < best_margin
            
            # Update best
            for b in range(batch_size):
                if improved[b]:
                    best_x_adv[b] = x_candidate[b]
                    best_margin[b] = cand_margin[b]
            
            successful = (best_margin < 0)
        
        return best_x_adv