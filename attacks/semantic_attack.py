"""Semantic adversarial attacks using interpretable transformations."""

from typing import Optional, List, Tuple, Literal, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_attack import BaseAttack, AttackResult


class SemanticAttack(BaseAttack):
    """Semantic adversarial attack using interpretable transformations.
    
    Instead of pixel-level noise, applies human-interpretable
    transformations like brightness, contrast, rotation, etc.
    
    References:
        - Hosseini & Poovendran, "Semantic Adversarial Examples", 2018
        - Engstrom et al., "Exploring the Landscape of Spatial Robustness", 2019
    """
    
    TRANSFORMS = ["brightness", "contrast", "saturation", "hue", "rotation", "translation", "scale"]
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.3,  # Semantic epsilon (transform magnitude)
        transforms: Optional[List[str]] = None,
        iterations: int = 50,
        step_size: float = 0.01,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
        # Transform-specific bounds
        max_brightness: float = 0.3,
        max_contrast: float = 0.3,
        max_saturation: float = 0.3,
        max_hue: float = 0.1,
        max_rotation: float = 15.0,  # degrees
        max_translation: float = 0.1,  # fraction
        max_scale: float = 0.1,  # fraction
    ):
        """Initialize semantic attack.
        
        Args:
            model: Target model
            epsilon: Global transform magnitude bound
            transforms: List of transforms to use
            iterations: Optimization iterations
            step_size: Gradient step size
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Dict with 'mean' and 'std' for image normalization
            max_*: Maximum magnitude for each transform
        """
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        
        # Default to spatial transforms (more effective for medical images)
        self.transforms = transforms if transforms else ["brightness", "contrast", "rotation"]
        self.iterations = iterations
        self.step_size = step_size
        
        # Transform bounds
        self.bounds = {
            "brightness": max_brightness,
            "contrast": max_contrast,
            "saturation": max_saturation,
            "hue": max_hue,
            "rotation": max_rotation,
            "translation": max_translation,
            "scale": max_scale,
        }
        
        # Validate transforms
        for t in self.transforms:
            if t not in self.TRANSFORMS:
                raise ValueError(f"Unknown transform: {t}. Available: {self.TRANSFORMS}")
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        transforms: Optional[List[str]] = None,
        iterations: Optional[int] = None,
        intensity: float = 1.0,
        **kwargs,
    ) -> AttackResult:
        """Perform semantic attack.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            target: Target labels for targeted attack
            transforms: Override transforms
            iterations: Override iterations
            intensity: Scale factor for transform magnitudes (0.1-1.0)
            
        Returns:
            AttackResult
        """
        trans = transforms if transforms else self.transforms
        iters = iterations if iterations else self.iterations
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # Scale bounds by intensity
        scaled_bounds = {k: v * intensity for k, v in self.bounds.items()}
        
        # Optimize transform parameters
        x_adv, params = self._optimize_transforms(x, y, target, trans, iters, scaled_bounds)
        
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
            epsilon_used=self.epsilon,
            strategy="semantic",
            metrics={"transform_params": params}
        )
    
    def _optimize_transforms(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        transforms: List[str],
        iterations: int,
        bounds: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Optimize transform parameters.
        
        Args:
            x: Input images
            y: True labels
            target: Target labels
            transforms: Transforms to use
            iterations: Number of iterations
            bounds: Optional custom bounds for transforms (scaled by intensity)
            
        Returns:
            (adversarial images, parameter dictionary)
        """
        # Use custom bounds or defaults
        active_bounds = bounds if bounds is not None else self.bounds
        
        batch_size = x.shape[0]
        
        # Initialize parameters for each transform
        params = {}
        for t in transforms:
            if t in ["brightness", "contrast", "saturation", "hue"]:
                params[t] = torch.zeros(batch_size, device=self.device, requires_grad=True)
            elif t == "rotation":
                params[t] = torch.zeros(batch_size, device=self.device, requires_grad=True)
            elif t == "translation":
                params[t] = torch.zeros(batch_size, 2, device=self.device, requires_grad=True)
            elif t == "scale":
                params[t] = torch.zeros(batch_size, device=self.device, requires_grad=True)
        
        # Optimizer for transform parameters - use higher LR for faster convergence
        param_list = [p for p in params.values()]
        optimizer = torch.optim.Adam(param_list, lr=self.step_size * 2)
        
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=self.step_size * 0.1)
        
        best_loss = float('inf')
        best_params = None
        
        for i in range(iterations):
            self._iteration_count = i + 1
            
            # Apply transforms
            x_adv = self._apply_transforms(x, params, transforms)
            
            # Compute loss
            logits = self.model(x_adv)
            self._query_count += batch_size
            
            if self.targeted and target is not None:
                loss = F.cross_entropy(logits, target)
            else:
                loss = -F.cross_entropy(logits, y)
            
            # Track best parameters
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {k: v.detach().clone() for k, v in params.items()}
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Project parameters to valid range (using potentially scaled bounds)
            with torch.no_grad():
                for t in transforms:
                    bound = active_bounds[t] * self.epsilon / 0.3  # Scale by epsilon
                    if t == "translation":
                        params[t].data = torch.clamp(params[t].data, -bound, bound)
                    else:
                        params[t].data = torch.clamp(params[t].data, -bound, bound)
            
            # Early stopping (check with projected perturbation)
            with torch.no_grad():
                x_adv_proj = self._apply_transforms(x, params, transforms)
                perturbation = x_adv_proj - x
                perturbation = self._project_perturbation(perturbation, self.epsilon)
                x_adv_proj = x + perturbation
                x_adv_proj = self._clamp_image(x_adv_proj)
                if self._check_success(x_adv_proj, y, target).all():
                    best_params = {k: v.detach().clone() for k, v in params.items()}
                    break
        
        # Use best parameters found
        if best_params is not None:
            params = best_params
        
        # Final adversarial example
        with torch.no_grad():
            x_adv = self._apply_transforms(x, params, transforms)
            
            # IMPORTANT: Project perturbation to respect epsilon L-inf bound
            perturbation = x_adv - x
            perturbation = self._project_perturbation(perturbation, self.epsilon)
            x_adv = x + perturbation
            x_adv = self._clamp_image(x_adv)
            
            params_final = {k: v.detach() for k, v in params.items()}
        
        return x_adv, params_final
    
    def _apply_transforms(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
        transforms: List[str],
    ) -> torch.Tensor:
        """Apply semantic transforms to images.
        
        Args:
            x: Input images
            params: Transform parameters
            transforms: List of transforms
            
        Returns:
            Transformed images
        """
        x_out = x.clone()
        
        for t in transforms:
            if t == "brightness":
                x_out = self._adjust_brightness(x_out, params[t])
            elif t == "contrast":
                x_out = self._adjust_contrast(x_out, params[t])
            elif t == "saturation":
                x_out = self._adjust_saturation(x_out, params[t])
            elif t == "hue":
                x_out = self._adjust_hue(x_out, params[t])
            elif t == "rotation":
                x_out = self._rotate(x_out, params[t])
            elif t == "translation":
                x_out = self._translate(x_out, params[t])
            elif t == "scale":
                x_out = self._scale(x_out, params[t])
        
        # Clamp to valid range (handles normalization)
        x_out = self._clamp_image(x_out)
        
        return x_out
    
    def _adjust_brightness(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust image brightness.
        
        Args:
            x: Images [B, C, H, W]
            delta: Brightness adjustment per sample [B]
            
        Returns:
            Adjusted images
        """
        return x + delta.view(-1, 1, 1, 1)
    
    def _adjust_contrast(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust image contrast.
        
        Args:
            x: Images [B, C, H, W]
            delta: Contrast adjustment per sample [B]
            
        Returns:
            Adjusted images
        """
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        factor = 1 + delta.view(-1, 1, 1, 1)
        return mean + factor * (x - mean)
    
    def _adjust_saturation(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust image saturation.
        
        Args:
            x: Images [B, C, H, W]
            delta: Saturation adjustment per sample [B]
            
        Returns:
            Adjusted images
        """
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gray = gray.expand_as(x)
        
        factor = 1 + delta.view(-1, 1, 1, 1)
        return gray + factor * (x - gray)
    
    def _adjust_hue(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust image hue (simplified).
        
        Args:
            x: Images [B, C, H, W]
            delta: Hue adjustment per sample [B]
            
        Returns:
            Adjusted images
        """
        # Simplified hue shift via channel rotation
        if x.shape[1] == 3:
            # Create rotation matrix for hue shift
            angle = delta * np.pi  # Convert to radians
            cos_a = torch.cos(angle).view(-1, 1, 1)
            sin_a = torch.sin(angle).view(-1, 1, 1)
            
            # Apply rotation in color space (simplified)
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            
            r_new = r * cos_a.unsqueeze(-1) - g * sin_a.unsqueeze(-1)
            g_new = r * sin_a.unsqueeze(-1) + g * cos_a.unsqueeze(-1)
            
            return torch.cat([r_new, g_new, b], dim=1)
        return x
    
    def _rotate(
        self,
        x: torch.Tensor,
        angle: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate images.
        
        Args:
            x: Images [B, C, H, W]
            angle: Rotation angle in degrees per sample [B]
            
        Returns:
            Rotated images
        """
        batch_size = x.shape[0]
        
        # Convert to radians
        theta = angle * np.pi / 180
        
        # Create rotation matrices
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Affine transformation matrix [B, 2, 3]
        affine = torch.zeros(batch_size, 2, 3, device=x.device)
        affine[:, 0, 0] = cos_t
        affine[:, 0, 1] = -sin_t
        affine[:, 1, 0] = sin_t
        affine[:, 1, 1] = cos_t
        
        # Apply transformation
        grid = F.affine_grid(affine, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    
    def _translate(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Translate images.
        
        Args:
            x: Images [B, C, H, W]
            offset: Translation offset (tx, ty) per sample [B, 2]
            
        Returns:
            Translated images
        """
        batch_size = x.shape[0]
        
        # Affine transformation matrix [B, 2, 3]
        affine = torch.zeros(batch_size, 2, 3, device=x.device)
        affine[:, 0, 0] = 1
        affine[:, 1, 1] = 1
        affine[:, 0, 2] = offset[:, 0]
        affine[:, 1, 2] = offset[:, 1]
        
        # Apply transformation
        grid = F.affine_grid(affine, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    
    def _scale(
        self,
        x: torch.Tensor,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        """Scale images.
        
        Args:
            x: Images [B, C, H, W]
            factor: Scale factor adjustment per sample [B]
            
        Returns:
            Scaled images
        """
        batch_size = x.shape[0]
        
        # Scale factor (1 + factor for slight zoom)
        scale = 1 + factor
        
        # Affine transformation matrix [B, 2, 3]
        affine = torch.zeros(batch_size, 2, 3, device=x.device)
        affine[:, 0, 0] = scale
        affine[:, 1, 1] = scale
        
        # Apply transformation
        grid = F.affine_grid(affine, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    
    def random_transform(
        self,
        x: torch.Tensor,
        magnitude: float = 0.1,
    ) -> torch.Tensor:
        """Apply random semantic transforms.
        
        Args:
            x: Input images
            magnitude: Transform magnitude
            
        Returns:
            Randomly transformed images
        """
        batch_size = x.shape[0]
        
        # Random parameters
        params = {}
        for t in self.transforms:
            bound = self.bounds[t] * magnitude
            if t == "translation":
                params[t] = torch.empty(batch_size, 2, device=self.device).uniform_(-bound, bound)
            else:
                params[t] = torch.empty(batch_size, device=self.device).uniform_(-bound, bound)
        
        return self._apply_transforms(x, params, self.transforms)