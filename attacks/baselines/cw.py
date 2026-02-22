"""Carlini & Wagner L2 attack baseline."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_attack import BaseAttack, AttackResult


class CarliniWagner(BaseAttack):
    """Carlini & Wagner L2 attack.
    
    Optimization-based attack that minimizes perturbation while
    ensuring misclassification.
    
    Reference:
        Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", 2017
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.5,  # L2 bound
        confidence: float = 0.0,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        binary_search_steps: int = 9,
        initial_c: float = 1e-3,
        targeted: bool = False,
        device: str = "cuda",
        normalize_stats: Optional[dict] = None,
    ):
        """Initialize C&W attack.
        
        Args:
            model: Target model
            epsilon: Maximum L2 perturbation
            confidence: Confidence parameter (kappa)
            learning_rate: Adam learning rate
            max_iterations: Maximum optimization iterations
            binary_search_steps: Binary search steps for c
            initial_c: Initial value of c
            targeted: Whether to perform targeted attack
            device: Compute device
            normalize_stats: Normalization statistics (mean, std)
        """
        super().__init__(model, epsilon, targeted, device, normalize_stats)
        
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_c = initial_c
    
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        confidence: Optional[float] = None,
        max_iterations: Optional[int] = None,
        **kwargs,
    ) -> AttackResult:
        """Perform C&W attack.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            target: Target labels for targeted attack
            confidence: Override confidence
            max_iterations: Override max iterations
            
        Returns:
            AttackResult
        """
        kappa = confidence if confidence is not None else self.confidence
        iters = max_iterations if max_iterations is not None else self.max_iterations
        
        self.reset_stats()
        
        x = x.to(self.device)
        y = y.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        batch_size = x.shape[0]
        
        # Convert to tanh space
        x_tanh = self._to_tanh_space(x)
        
        # Best adversarial examples
        best_x_adv = x.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Binary search for c
        c_lower = torch.zeros(batch_size, device=self.device)
        c_upper = torch.full((batch_size,), 1e10, device=self.device)
        c = torch.full((batch_size,), self.initial_c, device=self.device)
        
        for search_step in range(self.binary_search_steps):
            # Initialize perturbation in tanh space
            w = torch.zeros_like(x_tanh, requires_grad=True)
            optimizer = torch.optim.Adam([w], lr=self.learning_rate)
            
            prev_loss = float('inf')
            
            for i in range(iters):
                self._iteration_count = i + 1
                
                # Convert from tanh space
                x_adv = self._from_tanh_space(x_tanh + w)
                
                # Compute L2 distance
                l2_dist = ((x_adv - x) ** 2).view(batch_size, -1).sum(dim=1)
                
                # Compute logits
                logits = self.model(x_adv)
                self._query_count += batch_size
                
                # Compute f(x') for C&W
                if self.targeted and target is not None:
                    # Want target class to have highest logit
                    target_onehot = F.one_hot(target, logits.shape[1]).float()
                    real = (target_onehot * logits).sum(dim=1)
                    other = ((1 - target_onehot) * logits - target_onehot * 1e4).max(dim=1)[0]
                    f_loss = torch.clamp(other - real + kappa, min=0)
                else:
                    # Want true class to have lower logit than others
                    y_onehot = F.one_hot(y, logits.shape[1]).float()
                    real = (y_onehot * logits).sum(dim=1)
                    other = ((1 - y_onehot) * logits - y_onehot * 1e4).max(dim=1)[0]
                    f_loss = torch.clamp(real - other + kappa, min=0)
                
                # Total loss
                loss = l2_dist + c * f_loss
                total_loss = loss.sum()
                
                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Update best adversarial examples
                with torch.no_grad():
                    x_adv = self._from_tanh_space(x_tanh + w)
                    preds = logits.argmax(dim=1)
                    
                    if self.targeted and target is not None:
                        success = preds == target
                    else:
                        success = preds != y
                    
                    # Update if successful and smaller L2
                    l2_current = ((x_adv - x) ** 2).view(batch_size, -1).sum(dim=1).sqrt()
                    improve = success & (l2_current < best_l2)
                    
                    best_x_adv[improve] = x_adv[improve]
                    best_l2[improve] = l2_current[improve]
                
                # Early stopping
                if abs(total_loss.item() - prev_loss) < 1e-6:
                    break
                prev_loss = total_loss.item()
            
            # Update c using binary search
            with torch.no_grad():
                x_adv = self._from_tanh_space(x_tanh + w)
                preds = self.model(x_adv).argmax(dim=1)
                
                if self.targeted and target is not None:
                    success = preds == target
                else:
                    success = preds != y
                
                # Adjust c
                c_lower[success] = torch.max(c_lower[success], c[success])
                c_upper[~success] = torch.min(c_upper[~success], c[~success])
                
                c[success] = (c_lower[success] + c[success]) / 2
                c[~success] = (c[~success] + c_upper[~success]) / 2
        
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
            iterations=iters * self.binary_search_steps,
            queries=self._query_count,
            epsilon_used=best_l2.mean().item(),
            strategy="cw",
            metrics={"l2_norm": best_l2.mean().item()}
        )
    
    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [0, 1] to tanh space.
        
        Args:
            x: Images in [0, 1]
            
        Returns:
            Images in tanh space
        """
        # Clamp to avoid numerical issues
        x = torch.clamp(x, 1e-8, 1 - 1e-8)
        return torch.atanh(2 * x - 1)
    
    def _from_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from tanh space to [0, 1].
        
        Args:
            x: Images in tanh space
            
        Returns:
            Images in [0, 1]
        """
        return (torch.tanh(x) + 1) / 2
