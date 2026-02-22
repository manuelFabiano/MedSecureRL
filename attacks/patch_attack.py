"""Patch-based adversarial attack: localized L-inf PGD.

Unlike the original "adversarial patch" (Brown et al., 2017) which replaces
image content entirely, this attack applies L-inf bounded perturbations
only within a localized region. This respects the same epsilon budget as
PIXEL and FREQUENCY attacks, making comparison fair.

The key advantage: concentrating the perturbation budget on a small region
can be more effective than spreading it across the entire image, especially
when the model's decision depends on specific spatial regions.
"""

from typing import Optional, Tuple, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_attack import BaseAttack, AttackResult


class PatchAttack(BaseAttack):
    """Localized L-inf PGD attack.
    
    Applies PGD perturbation only within a rectangular patch region.
    The perturbation respects the same L-inf epsilon budget as global attacks.
    
    The RL agent controls:
    - epsilon: perturbation magnitude within the patch
    - iterations: number of PGD steps
    - step_size: gradient step size
    - patch_size: fraction of image dimension (0.1 = 10% of image)
    - position_x, position_y: normalized center position [0, 1]
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        patch_size: float = 0.15,
        iterations: int = 40,
        step_size: Optional[float] = None,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        
        self.patch_size = patch_size  # Fraction of image size
        self.iterations = iterations
        self.step_size = step_size if step_size is not None else 2.5 * epsilon / iterations
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        iterations: Optional[int] = None,
        step_size: Optional[float] = None,
        patch_size: Optional[float] = None,
        position_x: float = 0.5,
        position_y: float = 0.5,
        **kwargs,
    ) -> AttackResult:
        """Perform localized PGD attack.
        
        Args:
            x: Input images [B, C, H, W] (possibly normalized)
            y: True labels [B]
            target: Target labels for targeted attack
            epsilon: L-inf budget in pixel space [0,1]
            iterations: Number of PGD steps
            step_size: Step size per iteration
            patch_size: Patch size as fraction of image (0.0-1.0)
            position_x: Horizontal center of patch (0.0=left, 1.0=right)
            position_y: Vertical center of patch (0.0=top, 1.0=bottom)
            
        Returns:
            AttackResult
        """
        eps = epsilon if epsilon is not None else self.epsilon
        iters = iterations if iterations is not None else self.iterations
        alpha = step_size if step_size is not None else 2.5 * eps / max(iters, 1)
        p_size = patch_size if patch_size is not None else self.patch_size
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        batch_size, channels, height, width = x.shape
        
        # Compute patch region
        mask = self._create_mask(height, width, p_size, position_x, position_y)
        mask = mask.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
        
        # Run localized PGD
        x_adv = self._localized_pgd(x, y, target, eps, iters, alpha, mask)
        
        # Results
        perturbation = x_adv - x
        adv_preds = self._get_predictions(x_adv)
        success = self._check_success(x_adv, y, target)
        
        # Compute actual patch dimensions for metadata
        patch_h = int(height * p_size)
        patch_w = int(width * p_size)
        
        return AttackResult(
            original=x,
            adversarial=x_adv,
            perturbation=perturbation,
            original_label=y,
            adversarial_label=adv_preds,
            target_label=target,
            success=success.all().item(),
            iterations=iters,
            queries=self._query_count,
            epsilon_used=eps,
            strategy="patch",
            metrics={
                "patch_size": p_size,
                "patch_pixels": patch_h * patch_w,
                "image_pixels": height * width,
                "patch_coverage": (patch_h * patch_w) / (height * width),
                "position": (position_x, position_y),
            }
        )
    
    def _create_mask(
        self,
        height: int,
        width: int,
        patch_size_frac: float,
        pos_x: float,
        pos_y: float,
    ) -> torch.Tensor:
        """Create binary mask for patch region.
        
        Args:
            height, width: Image dimensions
            patch_size_frac: Patch size as fraction of min(H, W)
            pos_x: Horizontal center (0=left, 1=right)
            pos_y: Vertical center (0=top, 1=bottom)
            
        Returns:
            Binary mask [H, W]
        """
        # Patch dimensions
        patch_h = max(3, int(height * patch_size_frac))
        patch_w = max(3, int(width * patch_size_frac))
        
        # Patch center (clamped to keep patch within image)
        center_y = int(pos_y * height)
        center_x = int(pos_x * width)
        
        # Compute corners (ensure patch stays within image)
        top = max(0, center_y - patch_h // 2)
        left = max(0, center_x - patch_w // 2)
        bottom = min(height, top + patch_h)
        right = min(width, left + patch_w)
        
        # Adjust if patch was clipped at bottom/right edge
        if bottom - top < patch_h:
            top = max(0, bottom - patch_h)
        if right - left < patch_w:
            left = max(0, right - patch_w)
        
        mask = torch.zeros(height, width, device=self.device)
        mask[top:bottom, left:right] = 1.0
        
        return mask
    
    def _localized_pgd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        epsilon: float,
        iterations: int,
        step_size: float,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """PGD attack restricted to patch region.
        
        The perturbation is only applied within the mask region.
        L-inf projection ensures budget is respected in pixel space.
        
        Args:
            x: Input images (possibly normalized)
            y: True labels
            target: Target labels
            epsilon: L-inf budget in pixel space
            iterations: Number of PGD steps
            step_size: Step size per step (pixel space)
            mask: Binary mask [1, 1, H, W]
            
        Returns:
            Adversarial images
        """
        # Initialize perturbation (only within patch)
        eps_init = self.epsilon_normalized if self.normalize_stats is not None else epsilon
        delta = torch.empty_like(x).uniform_(-eps_init, eps_init)
        delta = delta * mask  # Zero out perturbation outside patch
        delta = self._project_perturbation(delta, epsilon)
        delta = delta * mask  # Re-apply mask after projection
        
        # Convert step_size to normalized space if needed
        alpha = step_size
        if self.normalize_stats is not None and self.std is not None:
            alpha = step_size / self.std.mean().item()
        
        for i in range(iterations):
            self._iteration_count = i + 1
            
            x_adv = x + delta
            x_adv = self._clamp_image(x_adv)
            
            # Compute gradient
            grad = self._compute_gradient(x_adv, y, target)
            
            # Mask gradient: only update within patch region
            grad = grad * mask
            
            # PGD step
            if self.targeted:
                delta = delta - alpha * grad.sign()
            else:
                delta = delta + alpha * grad.sign()
            
            # Apply mask and project
            delta = delta * mask
            delta = self._project_perturbation(delta, epsilon)
            delta = delta * mask  # Re-apply after projection
            
            # Check early stopping
            x_adv = x + delta
            x_adv = self._clamp_image(x_adv)
            if self._check_success(x_adv, y, target).all():
                break
        
        x_adv = x + delta
        x_adv = self._clamp_image(x_adv)
        
        return x_adv