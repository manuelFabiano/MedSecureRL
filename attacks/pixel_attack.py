"""Pixel-based adversarial attacks (FGSM, PGD, BIM)."""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn

from .base_attack import BaseAttack, AttackResult


class PixelAttack(BaseAttack):
    """Pixel-based adversarial attacks.
    
    Implements gradient-based perturbations in the pixel domain:
    - FGSM: Single-step attack
    - PGD: Multi-step attack with random initialization
    - BIM: Multi-step attack without random initialization
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        method: str = "pgd",
        iterations: int = 40,
        step_size: Optional[float] = None,
        random_start: bool = True,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
    ):
        """Initialize pixel attack.
        
        Args:
            model: Target model
            epsilon: Maximum L-inf perturbation
            method: Attack method ('fgsm', 'pgd', 'bim')
            iterations: Number of iterations (for iterative methods)
            step_size: Step size per iteration (default: epsilon/4)
            random_start: Random initialization for PGD
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Dict with 'mean' and 'std' for image normalization
        """
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        
        self.method = method.lower()
        self.iterations = iterations
        self.step_size = step_size if step_size is not None else epsilon / 4
        self.random_start = random_start
        
        if self.method not in ["fgsm", "pgd", "bim"]:
            raise ValueError(f"Unknown method: {method}. Use 'fgsm', 'pgd', or 'bim'")
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        iterations: Optional[int] = None,
        step_size: Optional[float] = None,
        momentum: float = 0.0,
        targeted: bool = False,
        random_start: bool = True,
        sparsity: float = 1.0,
        **kwargs,
    ) -> AttackResult:
        """Perform pixel-based attack with RL-controllable parameters.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            target: Target labels for targeted attack
            epsilon: Override epsilon
            iterations: Override iterations
            step_size: Override step size
            momentum: Momentum coefficient for gradient accumulation (0-0.95)
            targeted: If True, attack toward runner-up class
            random_start: Random initialization (True=PGD, False=BIM)
            sparsity: Fraction of pixels to perturb (1.0=all, 0.3=sparse)
            
        Returns:
            AttackResult
        """
        # Use overrides or defaults
        eps = epsilon if epsilon is not None else self.epsilon
        iters = iterations if iterations is not None else self.iterations
        alpha = step_size if step_size is not None else self.step_size
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Auto-select target for targeted attack
        if targeted and target is None:
            target = self._get_runner_up_class(x, y)
        if target is not None:
            target = target.to(self.device)
        
        # Compute saliency mask for sparse perturbation
        saliency_mask = None
        if sparsity < 1.0:
            saliency_mask = self._compute_saliency_mask(x, y, sparsity)
        
        # Dispatch to appropriate method
        if self.method == "fgsm":
            x_adv = self._fgsm(x, y, target, eps, saliency_mask)
            iters = 1
        else:
            # PGD or BIM with momentum and sparsity
            x_adv = self._pgd_advanced(
                x, y, target, eps, iters, alpha,
                random_start=random_start,
                momentum=momentum,
                saliency_mask=saliency_mask,
                is_targeted=targeted
            )
        
        # Compute perturbation
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
            iterations=iters,
            queries=self._query_count,
            epsilon_used=eps,
            strategy="pixel",
        )
    
    def _get_runner_up_class(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get second-most-likely class as target for targeted attack."""
        with torch.no_grad():
            logits = self.model(x)
            self._query_count += x.shape[0]
            
            # Mask out true class
            logits_masked = logits.clone()
            logits_masked.scatter_(1, y.unsqueeze(1), float('-inf'))
            
            return logits_masked.argmax(dim=1)
    
    def _compute_saliency_mask(
        self, x: torch.Tensor, y: torch.Tensor, sparsity: float
    ) -> torch.Tensor:
        """Compute mask to perturb only most salient pixels."""
        x_var = x.clone().requires_grad_(True)
        logits = self.model(x_var)
        loss = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
        loss.backward()
        
        grad_magnitude = x_var.grad.detach().abs()
        
        batch_size = x.shape[0]
        mask = torch.zeros_like(x)
        
        for i in range(batch_size):
            flat_grad = grad_magnitude[i].flatten()
            k = max(1, int(len(flat_grad) * sparsity))
            _, top_indices = flat_grad.topk(k)
            
            flat_mask = torch.zeros_like(flat_grad)
            flat_mask[top_indices] = 1.0
            mask[i] = flat_mask.view(x.shape[1:])
        
        return mask
    
    def _pgd_advanced(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        epsilon: float,
        iterations: int,
        step_size: float,
        random_start: bool = True,
        momentum: float = 0.0,
        saliency_mask: Optional[torch.Tensor] = None,
        is_targeted: bool = False,
    ) -> torch.Tensor:
        """Advanced PGD with momentum and sparse perturbation.
        
        Args:
            x: Input images
            y: True labels
            target: Target labels (for targeted attack)
            epsilon: Perturbation budget
            iterations: Number of iterations
            step_size: Step size per iteration
            random_start: Random initialization
            momentum: Momentum coefficient (0 = standard PGD)
            saliency_mask: Binary mask for sparse perturbation
            is_targeted: Whether this is a targeted attack
            
        Returns:
            Adversarial images
        """
        x_adv = x.clone()
        
        # Random initialization with proper scaling
        if random_start:
            noise = torch.empty_like(x).uniform_(-1, 1)
            if saliency_mask is not None:
                noise = noise * saliency_mask
            # Scale noise to epsilon budget in denormalized space
            noise = self._project_perturbation(noise * epsilon, epsilon)
            x_adv = x_adv + noise
            x_adv = self._clamp_image(x_adv)
        
        # Momentum buffer
        momentum_buffer = torch.zeros_like(x)
        
        # Convert step_size to appropriate scale
        # Use 2.5 * epsilon / iterations as standard PGD step size
        alpha = step_size
        if self.normalize_stats is not None and self.std is not None:
            # Scale step size to normalized space, but cap it
            alpha = min(step_size / self.std.mean().item(), 
                       2.5 * self.epsilon_normalized / max(iterations, 1))
        
        for i in range(iterations):
            self._iteration_count = i + 1
            
            # Compute gradient
            grad = self._compute_gradient(x_adv, y, target)
            
            # Apply saliency mask to gradient
            if saliency_mask is not None:
                grad = grad * saliency_mask
            
            # Always normalize gradient for stable updates
            grad_norm = grad / (grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8)
            
            # Momentum update
            if momentum > 0:
                momentum_buffer = momentum * momentum_buffer + grad_norm
                update_direction = momentum_buffer.sign()
            else:
                update_direction = grad_norm.sign()
            
            # Update adversarial example
            if is_targeted:
                # Move toward target class (minimize loss w.r.t target)
                x_adv = x_adv - alpha * update_direction
            else:
                # Move away from true class (maximize loss w.r.t true label)
                x_adv = x_adv + alpha * update_direction
            
            # Project back onto epsilon ball around x
            perturbation = x_adv - x
            if saliency_mask is not None:
                perturbation = perturbation * saliency_mask
            perturbation = self._project_perturbation(perturbation, epsilon)
            x_adv = x + perturbation
            
            # Clamp to valid image range
            x_adv = self._clamp_image(x_adv)
            
            # Early stopping if all samples are successful
            if self._check_success(x_adv, y, target).all():
                break
        
        return x_adv
    
    def _fgsm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        epsilon: float,
        saliency_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast Gradient Sign Method with optional sparse perturbation.
        
        Args:
            x: Input images
            y: True labels
            target: Target labels
            epsilon: Perturbation magnitude
            saliency_mask: Optional mask for sparse perturbation
            
        Returns:
            Adversarial images
        """
        grad = self._compute_gradient(x, y, target)
        
        # Apply saliency mask if provided
        if saliency_mask is not None:
            grad = grad * saliency_mask
        
        # Sign of gradient
        if self.targeted:
            # Move towards target
            perturbation = -epsilon * grad.sign()
        else:
            # Move away from true label
            perturbation = epsilon * grad.sign()
        
        if saliency_mask is not None:
            perturbation = perturbation * saliency_mask
        
        x_adv = x + perturbation
        x_adv = self._clamp_image(x_adv)
        
        self._iteration_count = 1
        return x_adv
    
    def _pgd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        epsilon: float,
        iterations: int,
        step_size: float,
        random_start: bool = True,
    ) -> torch.Tensor:
        """Projected Gradient Descent attack.
        
        Args:
            x: Input images
            y: True labels
            target: Target labels
            epsilon: Perturbation budget
            iterations: Number of iterations
            step_size: Step size per iteration
            random_start: Whether to use random initialization
            
        Returns:
            Adversarial images
        """
        x_adv = x.clone()
        
        # Random initialization
        if random_start:
            noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_adv = x_adv + noise
            x_adv = self._clamp_image(x_adv)
        
        for i in range(iterations):
            self._iteration_count = i + 1
            
            # Compute gradient
            grad = self._compute_gradient(x_adv, y, target)
            
            # Update adversarial example
            if self.targeted:
                x_adv = x_adv - step_size * grad.sign()
            else:
                x_adv = x_adv + step_size * grad.sign()
            
            # Project back onto epsilon ball around x
            perturbation = x_adv - x
            perturbation = self._project_perturbation(perturbation, epsilon)
            x_adv = x + perturbation
            
            # Clamp to valid image range
            x_adv = self._clamp_image(x_adv)
            
            # Early stopping if all samples are successful
            if self._check_success(x_adv, y, target).all():
                break
        
        return x_adv
    
    def get_perturbation_stats(
        self,
        perturbation: torch.Tensor,
    ) -> dict:
        """Get statistics about the perturbation.
        
        Args:
            perturbation: Perturbation tensor
            
        Returns:
            Dictionary with perturbation statistics
        """
        return {
            "linf": self.perturbation_norm(perturbation, "linf").mean().item(),
            "l2": self.perturbation_norm(perturbation, "l2").mean().item(),
            "l1": self.perturbation_norm(perturbation, "l1").mean().item(),
            "l0": self.perturbation_norm(perturbation, "l0").mean().item(),
        }


class AdaptivePixelAttack(PixelAttack):
    """Adaptive pixel attack with per-sample parameters.
    
    Extends PixelAttack to accept different epsilon/iterations
    for each sample in the batch, as selected by the RL agent.
    """
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[torch.Tensor] = None,
        iterations: Optional[torch.Tensor] = None,
        step_size: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform adaptive attack with per-sample parameters.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            target: Target labels
            epsilon: Per-sample epsilon [B] or scalar
            iterations: Per-sample iterations [B] or scalar
            step_size: Per-sample step size [B] or scalar
            
        Returns:
            AttackResult
        """
        # For now, use batch-wise parameters (take mean if tensor)
        if isinstance(epsilon, torch.Tensor):
            epsilon = epsilon.mean().item()
        if isinstance(iterations, torch.Tensor):
            iterations = int(iterations.max().item())
        if isinstance(step_size, torch.Tensor):
            step_size = step_size.mean().item()
        
        return super().attack(x, y, target, epsilon, iterations, step_size, **kwargs)