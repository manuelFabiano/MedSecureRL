"""AutoAttack wrapper for reliable evaluation.

Handles normalization correctly: AutoAttack operates in [0,1] pixel space,
and a wrapper model handles normalization internally.
"""

from typing import Optional, Literal, Dict, List
import torch
import torch.nn as nn

from ..base_attack import BaseAttack, AttackResult


class NormalizedModelWrapper(nn.Module):
    """Wraps a model that expects normalized input so it can receive [0,1] input.
    
    AutoAttack needs to work in [0,1] space for epsilon to be meaningful.
    This wrapper normalizes the input before passing it to the model.
    Also counts forward passes for query tracking.
    """
    def __init__(self, model: nn.Module, mean: List[float], std: List[float], device: str = "cuda"):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
        self.query_count = 0
    
    def reset_query_count(self):
        self.query_count = 0
    
    def forward(self, x):
        self.query_count += x.shape[0]
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


class AutoAttackWrapper(BaseAttack):
    """Wrapper for AutoAttack evaluation benchmark.
    
    Handles normalization properly: the model is wrapped so that AutoAttack
    operates in [0,1] pixel space, ensuring epsilon has consistent meaning
    with other attacks.
    
    Reference:
        Croce & Hein, "Reliable Evaluation of Adversarial Robustness with an 
        Ensemble of Diverse Parameter-free Attacks", ICML 2020
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        norm: Literal["Linf", "L2"] = "Linf",
        version: Literal["standard", "plus", "rand"] = "standard",
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
        verbose: bool = False,
    ):
        super().__init__(model, epsilon, targeted, device, normalize_stats=None)
        # Don't pass normalize_stats to BaseAttack - we handle it ourselves
        
        self.norm = norm
        self.version = version
        self.verbose = verbose
        self.normalize_stats_dict = normalize_stats
        
        # Create wrapped model that AutoAttack can use in [0,1] space
        if normalize_stats is not None:
            self._wrapped_model = NormalizedModelWrapper(
                model, 
                normalize_stats['mean'], 
                normalize_stats['std'],
                device
            ).to(device).eval()
        else:
            self._wrapped_model = model
        
        self._autoattack = None
    
    def _init_autoattack(self):
        """Initialize AutoAttack with the wrapped model."""
        try:
            from autoattack import AutoAttack
            
            self._autoattack = AutoAttack(
                self._wrapped_model,
                norm=self.norm,
                eps=self.epsilon,
                version='custom',  # Use custom to control which attacks run
                verbose=self.verbose,
            )
            # Only untargeted attacks — targeted APGD/FAB crash with few classes (e.g. 9)
            self._autoattack.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'square']
        except ImportError:
            raise ImportError(
                "AutoAttack not installed. Install with: pip install autoattack"
            )
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform AutoAttack.
        
        Args:
            x: Input images [B, C, H, W] (possibly normalized)
            y: True labels [B]
            target: Target labels (not supported)
            epsilon: Override epsilon (in [0,1] pixel space)
            
        Returns:
            AttackResult
        """
        if target is not None:
            raise NotImplementedError("Targeted AutoAttack not supported")
        
        eps = epsilon if epsilon is not None else self.epsilon
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Initialize AutoAttack if needed
        if self._autoattack is None or eps != self.epsilon:
            self.epsilon = eps
            self._init_autoattack()
        
        # Denormalize input to [0,1] for AutoAttack
        if self.normalize_stats_dict is not None:
            mean = torch.tensor(self.normalize_stats_dict['mean']).view(1, -1, 1, 1).to(self.device)
            std = torch.tensor(self.normalize_stats_dict['std']).view(1, -1, 1, 1).to(self.device)
            x_pixel = x * std + mean  # [0,1] space
        else:
            x_pixel = x
        
        # Reset query counter
        if hasattr(self._wrapped_model, 'reset_query_count'):
            self._wrapped_model.reset_query_count()
        
        # Run AutoAttack in [0,1] space
        x_adv_pixel = self._autoattack.run_standard_evaluation(x_pixel, y, bs=x.shape[0])
        
        # Get actual query count
        if hasattr(self._wrapped_model, 'query_count'):
            actual_queries = self._wrapped_model.query_count
        else:
            actual_queries = 0
        
        # Re-normalize adversarial back to model's normalized space
        if self.normalize_stats_dict is not None:
            x_adv = (x_adv_pixel - mean) / std
        else:
            x_adv = x_adv_pixel
        
        # Compute results
        perturbation = x_adv - x
        adv_preds = self._get_predictions(x_adv)
        success = self._check_success(x_adv, y, None)
        
        # Estimate query count from AutoAttack
        # Standard version: APGD-CE (100 iters) + APGD-DLR (100 iters) + FAB (100 iters) + Square (5000 queries)
        # Each iteration = 1 forward + 1 backward ≈ 2 queries (except Square which is query-based)
        # Rough estimate: ~5600 queries per sample for full standard eval
        # If attack succeeds early, fewer are used
        estimated_queries = 5600 if success.all().item() else 10000
        
        return AttackResult(
            original=x,
            adversarial=x_adv,
            perturbation=perturbation,
            original_label=y,
            adversarial_label=adv_preds,
            target_label=None,
            success=success.all().item(),
            iterations=-1,  # AutoAttack manages internally
            queries=actual_queries,
            epsilon_used=eps,
            strategy="autoattack",
        )