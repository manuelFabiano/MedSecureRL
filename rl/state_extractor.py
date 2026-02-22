"""State extraction for the RL environment.

Extracts a 21-dimensional state vector capturing:
- Gradient information (5 dims)
- Model confidence (4 dims)
- Attack progress (6 dims)
- Image characteristics (5 dims)
- Budget (1 dim)

Improvements over v1:
- Removed redundant features (eps_ratio = linf_ratio, strategy useless for separated SACs)
- Added confidence_drop and budget_usage for better attack progress tracking
- Fixed budget normalization for small epsilon ranges (log-scale)
- Fixed iter_progress to use actual max_steps
- Cached image features (static per episode, expensive to compute at 224x224)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StateExtractor:
    """Extract state features for the RL agent."""
    
    STATE_DIM = 21
    
    def __init__(self, model: nn.Module, device: str = "cuda", max_steps: int = 5):
        self.model = model
        self.device = device
        self.max_steps = max_steps
        
        # Running statistics for normalization
        self._mean = torch.zeros(self.STATE_DIM, device=device)
        self._var = torch.ones(self.STATE_DIM, device=device)
        self._count = 0
        
        # Training mode: if True, update running stats; if False, only normalize
        self.training_mode = True
        
        # Cache for static features (reset each episode)
        self._cached_image_features = None
        self._cached_original_confidence = None
        self._cache_valid = False
    
    def train(self):
        """Set to training mode (updates running statistics)."""
        self.training_mode = True
    
    def eval(self):
        """Set to eval mode (does not update running statistics)."""
        self.training_mode = False
    
    def reset_cache(self):
        """Reset cached features (call on each env.reset())."""
        self._cached_image_features = None
        self._cached_original_confidence = None
        self._cache_valid = False
    
    def save_stats(self, path: str):
        """Save normalization statistics."""
        torch.save({
            'mean': self._mean,
            'var': self._var,
            'count': self._count
        }, path)
    
    def load_stats(self, path: str):
        """Load normalization statistics."""
        import os
        if os.path.exists(path):
            stats = torch.load(path, map_location=self.device, weights_only=True)
            self._mean = stats['mean'].to(self.device)
            self._var = stats['var'].to(self.device)
            self._count = stats['count']
            return True
        return False
    
    def extract(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: Optional[torch.Tensor] = None,
        attack_iteration: int = 0,
        current_strategy: int = 0,
        epsilon: float = 0.03,
        epsilon_budget: float = 0.03,
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        batch_size = x.shape[0]
        if x_adv is None:
            x_adv = x
        
        grad_features = self._gradient_features(x_adv, y)        # 5
        conf_features = self._confidence_features(x_adv, y)       # 4
        attack_features = self._attack_features(                   # 6
            x, x_adv, y, attack_iteration, epsilon, epsilon_budget
        )
        
        # Image features: CACHED (static per episode)
        if not self._cache_valid:
            self._cached_image_features = self._image_features(x)
            with torch.no_grad():
                orig_logits = self.model(x)
                orig_probs = F.softmax(orig_logits, dim=1)
                self._cached_original_confidence = orig_probs.gather(1, y.unsqueeze(1)).item()
            self._cache_valid = True
        image_features = self._cached_image_features               # 5
        
        # Budget: log-scale for better resolution at small values
        # log(0.001)/log(0.1)=3, log(0.01)/log(0.1)=2, log(0.03)/log(0.1)=1.52
        budget_log = torch.full(
            (batch_size, 1),
            np.log(max(epsilon_budget, 1e-5)) / np.log(0.1),
            device=self.device
        )                                                          # 1
        
        state = torch.cat([
            grad_features, conf_features, attack_features,
            image_features, budget_log,
        ], dim=1)  # Total: 21
        
        return self._normalize(state, update_stats=self.training_mode)
    
    def _gradient_features(self, x, y):
        """Gradient-based features (5 dims): mean, max, std, sparsity, loss."""
        batch_size = x.shape[0]
        x_var = x.clone().detach().requires_grad_(True)
        logits = self.model(x_var)
        loss = F.cross_entropy(logits, y, reduction='none')
        loss.sum().backward()
        grad = x_var.grad.detach().clone()
        
        del x_var, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        grad_flat = grad.view(batch_size, -1)
        grad_mean = grad_flat.abs().mean(dim=1, keepdim=True)
        grad_max = grad_flat.abs().max(dim=1, keepdim=True)[0]
        grad_std = grad_flat.std(dim=1, keepdim=True)
        sparsity = (grad_flat.abs() < 0.01 * grad_max).float().mean(dim=1, keepdim=True)
        loss_norm = loss.detach().unsqueeze(1) / (loss.detach().max() + 1e-8)
        
        del grad, grad_flat, loss
        return torch.cat([grad_mean, grad_max, grad_std, sparsity, loss_norm], dim=1)
    
    def _confidence_features(self, x, y):
        """Confidence features (4 dims): max_prob, true_prob, entropy, margin."""
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)
            max_prob = probs.max(dim=1, keepdim=True)[0]
            true_prob = probs.gather(1, y.unsqueeze(1))
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=1, keepdim=True)
            entropy_norm = entropy / np.log(probs.shape[1])
            top2 = probs.topk(2, dim=1)[0]
            margin = top2[:, 0:1] - top2[:, 1:2]
        return torch.cat([max_prob, true_prob, entropy_norm, margin], dim=1)
    
    def _attack_features(self, x, x_adv, y, iteration, epsilon, epsilon_budget):
        """Attack progress features (6 dims).
        
        Replaces v1's redundant eps_ratio and useless strategy with:
        - confidence_drop: how much true-class prob dropped (progress signal)
        - budget_usage: epsilon_chosen / budget (is agent being conservative?)
        """
        batch_size = x.shape[0]
        pert = (x_adv - x).view(batch_size, -1)
        n_dims = float(np.prod(x.shape[1:]))
        
        linf = pert.abs().max(dim=1, keepdim=True)[0]
        l2 = pert.norm(dim=1, keepdim=True)
        l2_norm = l2 / (np.sqrt(n_dims) + 1e-8)
        
        # iter_progress: fixed to use actual max_steps
        iter_prog = torch.full((batch_size, 1), iteration / max(self.max_steps, 1), device=self.device)
        
        # confidence_drop: how much has true-class confidence fallen?
        orig_conf = self._cached_original_confidence if self._cached_original_confidence is not None else 1.0
        with torch.no_grad():
            curr_probs = F.softmax(self.model(x_adv), dim=1)
            curr_true_prob = curr_probs.gather(1, y.unsqueeze(1)).mean().item()
        conf_drop = torch.full((batch_size, 1), max(0.0, orig_conf - curr_true_prob), device=self.device)
        
        # budget_usage: how much of available budget did agent choose?
        budget_usage = torch.full((batch_size, 1), epsilon / (epsilon_budget + 1e-8), device=self.device)
        
        # perturbation efficiency: L2 / (Linf * sqrt(dims))
        efficiency = l2 / (linf * np.sqrt(n_dims) + 1e-8)
        
        return torch.cat([
            linf / (epsilon_budget + 1e-8),  # linf as fraction of budget
            l2_norm,
            iter_prog,
            conf_drop,
            budget_usage,
            efficiency,
        ], dim=1)
    
    def _image_features(self, x):
        """Image characteristics (5 dims). Computed once, cached. 
        
        mean_intensity, contrast, edge_density, texture_complexity, uniformity.
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        mean_intensity = x_flat.mean(dim=1, keepdim=True)
        contrast = x_flat.std(dim=1, keepdim=True)
        
        # Edge density (Sobel)
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=self.device).view(1,1,3,3)
        sobel_y = sobel_x.transpose(2, 3)
        edges = []
        for c in range(x.shape[1]):
            gx = F.conv2d(x[:, c:c+1], sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c+1], sobel_y, padding=1)
            edges.append((gx**2 + gy**2).sqrt().mean(dim=[1,2,3]))
        edge_density = torch.stack(edges, dim=1).mean(dim=1, keepdim=True)
        
        # Texture complexity (high-freq FFT energy)
        fft = torch.fft.fftshift(torch.fft.fft2(x))
        magnitude = torch.abs(fft)
        h, w = x.shape[2], x.shape[3]
        mask = torch.ones(h, w, device=self.device)
        mask[h//4:3*h//4, w//4:3*w//4] = 0
        total_energy = magnitude.view(batch_size, -1).sum(dim=1, keepdim=True)
        hf_energy = (magnitude * mask).view(batch_size, -1).sum(dim=1, keepdim=True)
        texture = hf_energy / (total_energy + 1e-8)
        
        # Spatial uniformity
        local_mean = F.avg_pool2d(x, 8, stride=4, padding=2)
        local_mean_up = F.interpolate(local_mean, size=x.shape[2:], mode='nearest')
        local_var = ((x - local_mean_up)**2).view(batch_size, -1).mean(dim=1, keepdim=True)
        uniformity = 1.0 / (local_var + 1e-3)
        uniformity = uniformity / (uniformity.max() + 1e-8)
        
        return torch.cat([mean_intensity, contrast, edge_density, texture, uniformity], dim=1)
    
    def _normalize(self, state, update_stats: bool = True):
        """Normalize using running statistics.
        
        Args:
            state: State tensor to normalize
            update_stats: If True, update running statistics (training mode).
                         If False, only normalize using existing stats (eval mode).
        """
        if update_stats:
            batch_mean = state.mean(dim=0)
            batch_var = state.var(dim=0, unbiased=False) if state.shape[0] > 1 else torch.zeros_like(batch_mean)
            
            self._count += state.shape[0]
            delta = batch_mean - self._mean
            self._mean = self._mean + delta * state.shape[0] / self._count
            # Use max with small value to prevent var from being too small
            self._var = torch.maximum(
                self._var + (batch_var - self._var) * state.shape[0] / self._count,
                torch.tensor(1e-4, device=self.device)
            )
        
        return (state - self._mean) / (self._var.sqrt() + 1e-8)
    
    def reset_normalization(self):
        self._mean = torch.zeros(self.STATE_DIM, device=self.device)
        self._var = torch.ones(self.STATE_DIM, device=self.device)
        self._count = 0
    
    def get_feature_names(self):
        return [
            "grad_mean", "grad_max", "grad_std", "grad_sparsity", "loss_normalized",
            "max_prob", "true_prob", "entropy", "margin",
            "linf_budget_ratio", "l2_normalized", "iter_progress",
            "confidence_drop", "budget_usage", "pert_efficiency",
            "mean_intensity", "contrast", "edge_density", "texture_complexity", "uniformity",
            "epsilon_budget_log",
        ]