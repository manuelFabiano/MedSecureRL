"""
Evaluation metrics for adversarial robustness testing.

Includes attack success rate, image quality metrics (SSIM, PSNR, LPIPS),
and perturbation metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not available. LPIPS metric will not work.")


@dataclass
class MetricResult:
    """Container for metric computation results."""
    name: str
    value: float
    std: Optional[float] = None
    per_sample: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


class AttackSuccessRate:
    """
    Compute Attack Success Rate (ASR) metrics.
    
    ASR measures the fraction of samples for which the attack
    successfully causes misclassification.
    """
    
    def __init__(self, targeted: bool = False):
        """
        Args:
            targeted: If True, measure success as reaching target class.
                     If False, measure success as leaving original class.
        """
        self.targeted = targeted
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total = 0
        self.successful = 0
        self.per_sample_success: List[bool] = []
    
    def update(
        self,
        original_preds: torch.Tensor,
        adversarial_preds: torch.Tensor,
        true_labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            original_preds: Predictions on clean images [batch_size]
            adversarial_preds: Predictions on adversarial images [batch_size]
            true_labels: Ground truth labels [batch_size]
            target_labels: Target labels for targeted attacks [batch_size]
        """
        batch_size = original_preds.shape[0]
        self.total += batch_size
        
        if self.targeted:
            if target_labels is None:
                raise ValueError("target_labels required for targeted ASR")
            # Success = adversarial prediction equals target
            success = (adversarial_preds == target_labels)
        else:
            # Success = adversarial prediction differs from original
            # Only count if original prediction was correct
            originally_correct = (original_preds == true_labels)
            misclassified = (adversarial_preds != true_labels)
            success = originally_correct & misclassified
        
        self.successful += success.sum().item()
        self.per_sample_success.extend(success.cpu().numpy().tolist())
    
    def compute(self) -> MetricResult:
        """Compute the ASR metric."""
        if self.total == 0:
            return MetricResult(
                name="ASR" if not self.targeted else "Targeted_ASR",
                value=0.0,
                std=0.0
            )
        
        asr = self.successful / self.total
        per_sample = np.array(self.per_sample_success)
        std = np.std(per_sample) if len(per_sample) > 1 else 0.0
        
        return MetricResult(
            name="ASR" if not self.targeted else "Targeted_ASR",
            value=asr,
            std=std,
            per_sample=per_sample,
            metadata={
                "total": self.total,
                "successful": self.successful,
                "targeted": self.targeted
            }
        )


class SSIM:
    """
    Structural Similarity Index Measure (SSIM).
    
    Measures structural similarity between original and adversarial images.
    Higher values indicate more similar images (less perceptible perturbation).
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channel: int = 3,
        size_average: bool = True,
        normalize_stats: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian window
            data_range: Range of input data (1.0 for [0,1], 255 for [0,255])
            channel: Number of channels
            size_average: If True, return mean over batch
            normalize_stats: Dict with 'mean' and 'std' if images are normalized
        """
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = channel
        self.size_average = size_average
        self.normalize_stats = normalize_stats
        self.window = self._create_window(window_size, channel, sigma)
        self.reset()
    
    def _create_window(
        self,
        window_size: int,
        channel: int,
        sigma: float
    ) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def reset(self):
        """Reset accumulated statistics."""
        self.values: List[float] = []
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize images to [0,1] range for metric computation."""
        if self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(x.device)
            std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(x.device)
            return x * std + mean
        return x
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SSIM between two batches of images."""
        device = img1.device
        window = self.window.to(device)
        channel = img1.shape[1]
        
        if channel != self.channel:
            window = self._create_window(self.window_size, channel, self.sigma)
            window = window.to(device)
        
        padding = self.window_size // 2
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.conv2d(img1 ** 2, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
        
        # Constants for stability
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Mean over spatial dimensions
        ssim_per_image = ssim_map.mean(dim=[1, 2, 3])
        
        return ssim_map, ssim_per_image
    
    def update(self, original: torch.Tensor, adversarial: torch.Tensor):
        """
        Update metrics with a batch of image pairs.
        
        Args:
            original: Original images [batch_size, C, H, W] (possibly normalized)
            adversarial: Adversarial images [batch_size, C, H, W] (possibly normalized)
        """
        # Denormalize for correct SSIM computation
        original_denorm = self._denormalize(original)
        adversarial_denorm = self._denormalize(adversarial)
        
        _, ssim_values = self._ssim(original_denorm, adversarial_denorm)
        self.values.extend(ssim_values.cpu().numpy().tolist())
    
    def compute(self) -> MetricResult:
        """Compute mean SSIM."""
        if len(self.values) == 0:
            return MetricResult(name="SSIM", value=0.0)
        
        values = np.array(self.values)
        return MetricResult(
            name="SSIM",
            value=float(np.mean(values)),
            std=float(np.std(values)),
            per_sample=values
        )


class PSNR:
    """
    Peak Signal-to-Noise Ratio (PSNR).
    
    Measures quality of adversarial perturbation.
    Higher values indicate less perceptible perturbation.
    """
    
    def __init__(
        self,
        data_range: float = 1.0,
        eps: float = 1e-10,
        normalize_stats: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            data_range: Range of input data (1.0 for [0,1], 255 for [0,255])
            eps: Small constant for numerical stability
            normalize_stats: Dict with 'mean' and 'std' if images are normalized
        """
        self.data_range = data_range
        self.eps = eps
        self.normalize_stats = normalize_stats
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.values: List[float] = []
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize images to [0,1] range for metric computation."""
        if self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(x.device)
            std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(x.device)
            return x * std + mean
        return x
    
    def update(self, original: torch.Tensor, adversarial: torch.Tensor):
        """
        Update metrics with a batch of image pairs.
        
        Args:
            original: Original images [batch_size, C, H, W] (possibly normalized)
            adversarial: Adversarial images [batch_size, C, H, W] (possibly normalized)
        """
        # Denormalize for correct PSNR computation
        original_denorm = self._denormalize(original)
        adversarial_denorm = self._denormalize(adversarial)
        
        # Compute MSE per image
        mse = ((original_denorm - adversarial_denorm) ** 2).mean(dim=[1, 2, 3])
        
        # PSNR formula
        psnr = 10 * torch.log10((self.data_range ** 2) / (mse + self.eps))
        
        self.values.extend(psnr.cpu().numpy().tolist())
    
    def compute(self) -> MetricResult:
        """Compute mean PSNR."""
        if len(self.values) == 0:
            return MetricResult(name="PSNR", value=0.0)
        
        values = np.array(self.values)
        return MetricResult(
            name="PSNR",
            value=float(np.mean(values)),
            std=float(np.std(values)),
            per_sample=values,
            metadata={"unit": "dB"}
        )


class LPIPS_Metric:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    
    Uses deep features to measure perceptual similarity.
    Lower values indicate more similar images.
    """
    
    def __init__(
        self,
        net: str = 'alex',
        device: Optional[torch.device] = None,
        normalize_stats: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            device: Device for computation
            normalize_stats: Dict with 'mean' and 'std' if images are normalized
        """
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips package not installed. Run: pip install lpips")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net
        self.normalize_stats = normalize_stats
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.values: List[float] = []
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize images to [0,1] range for metric computation."""
        if self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(x.device)
            std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(x.device)
            return x * std + mean
        return x
    
    @torch.no_grad()
    def update(self, original: torch.Tensor, adversarial: torch.Tensor):
        """
        Update metrics with a batch of image pairs.
        
        Args:
            original: Original images [batch_size, C, H, W] (possibly normalized)
            adversarial: Adversarial images [batch_size, C, H, W] (possibly normalized)
        """
        # Denormalize first
        original_denorm = self._denormalize(original)
        adversarial_denorm = self._denormalize(adversarial)
        
        # LPIPS expects images in [-1, 1]
        original_scaled = 2 * original_denorm.to(self.device) - 1
        adversarial_scaled = 2 * adversarial_denorm.to(self.device) - 1
        
        # LPIPS requires minimum image size (64x64) for VGG network
        _, _, H, W = original_scaled.shape
        if H < 64 or W < 64:
            import torch.nn.functional as F
            target_size = (max(64, H), max(64, W))
            original_scaled = F.interpolate(original_scaled, size=target_size, mode='bilinear', align_corners=False)
            adversarial_scaled = F.interpolate(adversarial_scaled, size=target_size, mode='bilinear', align_corners=False)
        
        # Compute LPIPS distance
        distances = self.loss_fn(original_scaled, adversarial_scaled)
        
        # Handle both single image and batch cases
        distances_np = distances.cpu().numpy().flatten()
        self.values.extend(distances_np.tolist())
    
    def compute(self) -> MetricResult:
        """Compute mean LPIPS distance."""
        if len(self.values) == 0:
            return MetricResult(name="LPIPS", value=0.0)
        
        values = np.array(self.values)
        return MetricResult(
            name="LPIPS",
            value=float(np.mean(values)),
            std=float(np.std(values)),
            per_sample=values,
            metadata={"network": self.net}
        )


class PerturbationMetrics:
    """
    Compute various perturbation norm metrics.
    """
    
    def __init__(self, normalize_stats: Optional[Dict[str, List[float]]] = None):
        """
        Args:
            normalize_stats: Dict with 'mean' and 'std' if images are normalized
        """
        self.normalize_stats = normalize_stats
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.l_inf: List[float] = []
        self.l_2: List[float] = []
        self.l_1: List[float] = []
        self.l_0: List[float] = []
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize images to [0,1] range for metric computation."""
        if self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(x.device)
            std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(x.device)
            return x * std + mean
        return x
    
    def update(self, original: torch.Tensor, adversarial: torch.Tensor):
        """
        Update metrics with a batch of image pairs.
        
        Args:
            original: Original images [batch_size, C, H, W] (possibly normalized)
            adversarial: Adversarial images [batch_size, C, H, W] (possibly normalized)
        """
        # Denormalize for correct metric computation in [0,1] space
        original_denorm = self._denormalize(original)
        adversarial_denorm = self._denormalize(adversarial)
        
        perturbation = adversarial_denorm - original_denorm
        batch_size = perturbation.shape[0]
        
        # Flatten spatial dimensions
        flat_pert = perturbation.view(batch_size, -1)
        
        # L-infinity norm
        l_inf = flat_pert.abs().max(dim=1)[0]
        self.l_inf.extend(l_inf.cpu().numpy().tolist())
        
        # L2 norm
        l_2 = torch.norm(flat_pert, p=2, dim=1)
        self.l_2.extend(l_2.cpu().numpy().tolist())
        
        # L1 norm
        l_1 = torch.norm(flat_pert, p=1, dim=1)
        self.l_1.extend(l_1.cpu().numpy().tolist())
        
        # L0 norm (number of changed pixels)
        l_0 = (flat_pert.abs() > 1e-8).sum(dim=1).float()
        self.l_0.extend(l_0.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, MetricResult]:
        """Compute all perturbation metrics."""
        results = {}
        
        for name, values in [
            ("L_inf", self.l_inf),
            ("L_2", self.l_2),
            ("L_1", self.l_1),
            ("L_0", self.l_0)
        ]:
            if len(values) == 0:
                results[name] = MetricResult(name=name, value=0.0)
            else:
                arr = np.array(values)
                results[name] = MetricResult(
                    name=name,
                    value=float(np.mean(arr)),
                    std=float(np.std(arr)),
                    per_sample=arr
                )
        
        return results


class ComprehensiveMetrics:
    """
    Unified interface for computing all evaluation metrics.
    """
    
    def __init__(
        self,
        compute_lpips: bool = True,
        lpips_net: str = 'alex',
        targeted: bool = False,
        device: Optional[torch.device] = None,
        normalize_stats: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            compute_lpips: Whether to compute LPIPS (requires lpips package)
            lpips_net: Network for LPIPS
            targeted: Whether attacks are targeted
            device: Device for computation
            normalize_stats: Normalization statistics for correct metric computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # For PathMNIST/ImageNet normalization - use as default if not provided
        if normalize_stats is None:
            normalize_stats = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        
        self.asr = AttackSuccessRate(targeted=targeted)
        self.ssim = SSIM(normalize_stats=normalize_stats)
        self.psnr = PSNR(normalize_stats=normalize_stats)
        self.perturbation = PerturbationMetrics(normalize_stats=normalize_stats)
        
        self.compute_lpips_flag = compute_lpips and LPIPS_AVAILABLE
        if self.compute_lpips_flag:
            self.lpips = LPIPS_Metric(net=lpips_net, device=self.device, normalize_stats=normalize_stats)
        else:
            self.lpips = None
    
    def reset(self):
        """Reset all metrics."""
        self.asr.reset()
        self.ssim.reset()
        self.psnr.reset()
        self.perturbation.reset()
        if self.lpips is not None:
            self.lpips.reset()
    
    def update(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        original_preds: torch.Tensor,
        adversarial_preds: torch.Tensor,
        true_labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ):
        """
        Update all metrics with a batch.
        
        Args:
            original: Original images [batch_size, C, H, W]
            adversarial: Adversarial images [batch_size, C, H, W]
            original_preds: Predictions on clean images [batch_size]
            adversarial_preds: Predictions on adversarial images [batch_size]
            true_labels: Ground truth labels [batch_size]
            target_labels: Target labels for targeted attacks [batch_size]
        """
        self.asr.update(original_preds, adversarial_preds, true_labels, target_labels)
        self.ssim.update(original, adversarial)
        self.psnr.update(original, adversarial)
        self.perturbation.update(original, adversarial)
        
        if self.lpips is not None:
            self.lpips.update(original, adversarial)
    
    def compute(self) -> Dict[str, MetricResult]:
        """Compute all metrics."""
        results = {
            "ASR": self.asr.compute(),
            "SSIM": self.ssim.compute(),
            "PSNR": self.psnr.compute(),
        }
        
        # Add perturbation metrics
        results.update(self.perturbation.compute())
        
        if self.lpips is not None:
            results["LPIPS"] = self.lpips.compute()
        
        return results
    
    def summary(self) -> Dict[str, float]:
        """Get summary of all metrics as simple dict."""
        results = self.compute()
        return {name: result.value for name, result in results.items()}


def compute_fooling_rate(
    model: torch.nn.Module,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    true_labels: torch.Tensor
) -> float:
    """
    Compute fooling rate: fraction of samples where adversarial
    example causes different prediction than original.
    
    Args:
        model: Target model
        original: Original images
        adversarial: Adversarial images
        true_labels: Ground truth labels
    
    Returns:
        Fooling rate in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        original_preds = model(original).argmax(dim=1)
        adversarial_preds = model(adversarial).argmax(dim=1)
    
    fooled = (original_preds != adversarial_preds)
    return fooled.float().mean().item()


def compute_transfer_rate(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    true_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute transferability of adversarial examples.
    
    Args:
        source_model: Model used to generate adversarial examples
        target_model: Model to test transferability
        original: Original images
        adversarial: Adversarial images
        true_labels: Ground truth labels
    
    Returns:
        Dictionary with source ASR, target ASR, and transfer rate
    """
    source_model.eval()
    target_model.eval()
    
    with torch.no_grad():
        # Source model predictions
        source_orig_preds = source_model(original).argmax(dim=1)
        source_adv_preds = source_model(adversarial).argmax(dim=1)
        
        # Target model predictions
        target_orig_preds = target_model(original).argmax(dim=1)
        target_adv_preds = target_model(adversarial).argmax(dim=1)
    
    # Source ASR (should be high)
    source_correct = (source_orig_preds == true_labels)
    source_fooled = (source_adv_preds != true_labels)
    source_asr = (source_correct & source_fooled).float().mean().item()
    
    # Target ASR (transfer success)
    target_correct = (target_orig_preds == true_labels)
    target_fooled = (target_adv_preds != true_labels)
    target_asr = (target_correct & target_fooled).float().mean().item()
    
    # Transfer rate among successful source attacks
    successful_source = source_correct & source_fooled
    if successful_source.sum() > 0:
        transfer_rate = (successful_source & target_fooled).sum().item() / successful_source.sum().item()
    else:
        transfer_rate = 0.0
    
    return {
        "source_asr": source_asr,
        "target_asr": target_asr,
        "transfer_rate": transfer_rate
    }