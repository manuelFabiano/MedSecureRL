"""Frequency-domain adversarial attacks using low-frequency perturbations.

This implementation uses a correct approach:
- Perturbation is stored in SPATIAL domain
- Gradient is filtered in FREQUENCY domain (low-pass)
- Projection is done in SPATIAL domain

This ensures the epsilon budget is respected while encouraging low-frequency perturbations.
"""

from typing import Optional, Tuple, Literal, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_attack import BaseAttack, AttackResult


class FrequencyAttack(BaseAttack):
    """Frequency-domain adversarial attack using low-frequency perturbations.
    
    This attack encourages perturbations in specific frequency bands by
    filtering the gradient before each update step. The perturbation itself
    is stored and projected in the spatial domain to ensure the epsilon
    budget is correctly enforced.
    
    References:
        - Guo et al., "Low Frequency Adversarial Perturbation", ICML 2019
        - Sharma et al., "On the Effectiveness of Low Frequency Perturbations", IJCAI 2019
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        target_bands: Literal["low", "mid", "high", "all", "adaptive"] = "low",
        band_ratio: float = 0.25,
        iterations: int = 40,
        step_size: Optional[float] = None,
        random_start: bool = True,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
    ):
        """Initialize frequency attack.
        
        Args:
            model: Target model
            epsilon: Maximum L-inf perturbation magnitude
            target_bands: Which frequency bands to use for perturbation
                - "low": Low frequencies only (smoother perturbations)
                - "mid": Mid frequencies
                - "high": High frequencies (edge-like perturbations)
                - "all" or "adaptive": All frequencies (equivalent to standard PGD)
            band_ratio: Fraction of spectrum to include (0.0 to 1.0)
            iterations: Number of optimization iterations
            step_size: Gradient step size (default: 2.5 * epsilon / iterations)
            random_start: Whether to use random initialization
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Dict with 'mean' and 'std' for image normalization
        """
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        
        # Treat "adaptive" as alias for "all"
        self.target_bands = "all" if target_bands == "adaptive" else target_bands
        self.band_ratio = band_ratio
        self.iterations = iterations
        self.step_size = step_size if step_size is not None else 2.5 * epsilon / iterations
        self.random_start = random_start
        
        # Cache for frequency masks (to avoid recomputing)
        self._mask_cache = {}
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        iterations: Optional[int] = None,
        step_size: Optional[float] = None,
        band_ratio: Optional[float] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform frequency-domain attack.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            target: Target labels for targeted attack
            epsilon: Override epsilon
            iterations: Override iterations
            step_size: Override step size
            band_ratio: Override band ratio
            
        Returns:
            AttackResult with adversarial examples
        """
        eps = epsilon if epsilon is not None else self.epsilon
        iters = iterations if iterations is not None else self.iterations
        alpha = step_size if step_size is not None else 2.5 * eps / iters
        ratio = band_ratio if band_ratio is not None else self.band_ratio
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # Create frequency mask for gradient filtering
        mask = self._get_frequency_mask(x.shape[-2:], ratio)
        
        # Perform attack
        x_adv = self._low_freq_pgd(x, y, target, eps, iters, alpha, mask)
        
        # Compute results
        perturbation = x_adv - x
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
            strategy="frequency",
        )
    
    def _get_frequency_mask(self, shape: Tuple[int, int], ratio: float) -> torch.Tensor:
        """Get or create frequency mask for the given shape.
        
        Args:
            shape: (H, W) image dimensions
            ratio: Band ratio
            
        Returns:
            Frequency mask [H, W]
        """
        cache_key = (shape, ratio, self.target_bands)
        
        if cache_key not in self._mask_cache:
            h, w = shape
            
            # Create frequency coordinate grid (shifted so DC is at center)
            freq_y = torch.fft.fftfreq(h, device=self.device)
            freq_x = torch.fft.fftfreq(w, device=self.device)
            freq_yy, freq_xx = torch.meshgrid(freq_y, freq_x, indexing='ij')
            
            # Normalized distance from DC (0 to ~0.7 for corners)
            dist = torch.sqrt(freq_yy**2 + freq_xx**2)
            max_dist = torch.sqrt(torch.tensor(0.5**2 + 0.5**2))  # Max possible
            dist_normalized = dist / max_dist
            
            # Create mask based on target bands
            if self.target_bands == "low":
                # Low frequencies: close to DC
                mask = (dist_normalized <= ratio).float()
            elif self.target_bands == "high":
                # High frequencies: far from DC
                mask = (dist_normalized >= (1 - ratio)).float()
            elif self.target_bands == "mid":
                # Mid frequencies
                low_bound = 0.5 - ratio / 2
                high_bound = 0.5 + ratio / 2
                mask = ((dist_normalized >= low_bound) & (dist_normalized <= high_bound)).float()
            else:  # "all"
                mask = torch.ones(h, w, device=self.device)
            
            # Ensure DC component is always included for stability
            mask[0, 0] = 1.0
            
            self._mask_cache[cache_key] = mask
        
        return self._mask_cache[cache_key]
    
    def _filter_gradient(
        self,
        grad: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Filter gradient in frequency domain.
        
        Args:
            grad: Spatial gradient [B, C, H, W]
            mask: Frequency mask [H, W]
            
        Returns:
            Filtered gradient [B, C, H, W]
        """
        # Transform to frequency domain
        grad_freq = torch.fft.fft2(grad)
        
        # Apply mask (broadcast over batch and channel dims)
        grad_freq_filtered = grad_freq * mask.unsqueeze(0).unsqueeze(0)
        
        # Transform back to spatial domain
        grad_filtered = torch.fft.ifft2(grad_freq_filtered).real
        
        return grad_filtered
    
    def _low_freq_pgd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor],
        epsilon: float,
        iterations: int,
        step_size: float,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """PGD attack with low-frequency gradient filtering.
        
        The key insight is:
        - Perturbation is stored in SPATIAL domain
        - Gradient is filtered to encourage low-frequency updates
        - L-inf projection happens in SPATIAL domain
        
        Args:
            x: Input images
            y: True labels
            target: Target labels
            epsilon: Perturbation budget (L-inf)
            iterations: Number of iterations
            step_size: Step size per iteration
            mask: Frequency mask for gradient filtering
            
        Returns:
            Adversarial images
        """
        # Initialize perturbation in spatial domain
        # Use normalized epsilon if image is normalized
        eps_init = self.epsilon_normalized if self.normalize_stats is not None else epsilon
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-eps_init, eps_init)
        else:
            delta = torch.zeros_like(x)
        
        delta = self._project_perturbation(delta, epsilon)
        
        # Convert step_size to normalized space if needed
        alpha = step_size
        if self.normalize_stats is not None and self.std is not None:
            alpha = step_size / self.std.mean().item()
        
        for i in range(iterations):
            self._iteration_count = i + 1
            
            # Current adversarial example
            x_adv = x + delta
            x_adv = self._clamp_image(x_adv)
            
            # Compute gradient
            grad = self._compute_gradient(x_adv, y, target)
            
            # Filter gradient in frequency domain
            grad_filtered = self._filter_gradient(grad, mask)
            
            # Update perturbation using filtered gradient
            if self.targeted:
                delta = delta - alpha * grad_filtered.sign()
            else:
                delta = delta + alpha * grad_filtered.sign()
            
            # Project to epsilon ball (L-inf)
            delta = self._project_perturbation(delta, epsilon)
            
            # Early stopping if successful
            x_adv = x + delta
            x_adv = self._clamp_image(x_adv)
            if self._check_success(x_adv, y, target).all():
                break
        
        # Final adversarial example
        x_adv = x + delta
        x_adv = self._clamp_image(x_adv)
        
        return x_adv
    
    def get_frequency_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Get frequency spectrum magnitude for visualization.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Log magnitude spectrum (shifted so DC is at center)
        """
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-10)
        return magnitude
    
    def visualize_mask(self, shape: Tuple[int, int], ratio: float = None) -> torch.Tensor:
        """Visualize the frequency mask.
        
        Args:
            shape: (H, W) dimensions
            ratio: Band ratio (uses default if None)
            
        Returns:
            Mask tensor [H, W] (shifted for visualization)
        """
        ratio = ratio if ratio is not None else self.band_ratio
        mask = self._get_frequency_mask(shape, ratio)
        # Shift for visualization (DC at center)
        return torch.fft.fftshift(mask)