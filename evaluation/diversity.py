"""
Diversity metrics for measuring vulnerability coverage.

Evaluates how well the RL agent explores different attack strategies
and discovers diverse vulnerabilities in the target model.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class DiversityResult:
    """Container for diversity metric results."""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class StrategyDiversity:
    """
    Measures diversity of attack strategies used.
    
    Tracks which attack strategies (pixel, frequency, patch, semantic)
    are successfully employed and their distribution.
    """
    
    STRATEGIES = ['pixel', 'frequency', 'patch', 'semantic']
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.strategy_counts: Dict[str, int] = defaultdict(int)
        self.successful_by_strategy: Dict[str, int] = defaultdict(int)
        self.total_attacks = 0
        self.total_successful = 0
    
    def update(
        self,
        strategy: str,
        success: bool
    ):
        """
        Record an attack attempt.
        
        Args:
            strategy: Attack strategy used ('pixel', 'frequency', etc.)
            success: Whether attack was successful
        """
        self.strategy_counts[strategy] += 1
        self.total_attacks += 1
        
        if success:
            self.successful_by_strategy[strategy] += 1
            self.total_successful += 1
    
    def update_batch(
        self,
        strategies: List[str],
        successes: List[bool]
    ):
        """Update with a batch of attacks."""
        for strategy, success in zip(strategies, successes):
            self.update(strategy, success)
    
    def compute_coverage(self) -> float:
        """
        Compute strategy coverage: fraction of strategies used successfully.
        
        Returns:
            Coverage in [0, 1]
        """
        strategies_used = set(self.successful_by_strategy.keys())
        return len(strategies_used) / len(self.STRATEGIES)
    
    def compute_entropy(self) -> float:
        """
        Compute entropy of strategy distribution.
        
        Higher entropy means more uniform distribution across strategies.
        
        Returns:
            Entropy in [0, log(n_strategies)]
        """
        if self.total_attacks == 0:
            return 0.0
        
        probs = []
        for strategy in self.STRATEGIES:
            count = self.strategy_counts.get(strategy, 0)
            if count > 0:
                probs.append(count / self.total_attacks)
        
        if len(probs) == 0:
            return 0.0
        
        probs = np.array(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def compute_normalized_entropy(self) -> float:
        """
        Compute normalized entropy (0 = single strategy, 1 = uniform).
        
        Returns:
            Normalized entropy in [0, 1]
        """
        max_entropy = np.log(len(self.STRATEGIES))
        return self.compute_entropy() / max_entropy if max_entropy > 0 else 0.0
    
    def compute_per_strategy_asr(self) -> Dict[str, float]:
        """
        Compute ASR for each strategy.
        
        Returns:
            Dictionary mapping strategy to ASR
        """
        results = {}
        for strategy in self.STRATEGIES:
            total = self.strategy_counts.get(strategy, 0)
            successful = self.successful_by_strategy.get(strategy, 0)
            results[strategy] = successful / total if total > 0 else 0.0
        return results
    
    def compute(self) -> DiversityResult:
        """Compute all diversity metrics."""
        return DiversityResult(
            name="StrategyDiversity",
            value=self.compute_normalized_entropy(),
            details={
                "coverage": self.compute_coverage(),
                "entropy": self.compute_entropy(),
                "normalized_entropy": self.compute_normalized_entropy(),
                "per_strategy_asr": self.compute_per_strategy_asr(),
                "strategy_counts": dict(self.strategy_counts),
                "successful_by_strategy": dict(self.successful_by_strategy),
                "total_attacks": self.total_attacks,
                "total_successful": self.total_successful
            }
        )


class PerturbationDiversity:
    """
    Measures diversity in the perturbation space.
    
    Tracks how different the generated perturbations are from each other.
    """
    
    def __init__(self, n_bins: int = 20):
        """
        Args:
            n_bins: Number of bins for histogram-based diversity
        """
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.perturbations: List[torch.Tensor] = []
        self.magnitudes: List[float] = []
        self.directions: List[torch.Tensor] = []
    
    def update(self, perturbation: torch.Tensor):
        """
        Add a perturbation to the collection.
        
        Args:
            perturbation: Perturbation tensor [C, H, W] or [batch, C, H, W]
        """
        if perturbation.dim() == 4:
            # Batch of perturbations
            for i in range(perturbation.shape[0]):
                self._add_single(perturbation[i])
        else:
            self._add_single(perturbation)
    
    def _add_single(self, perturbation: torch.Tensor):
        """Add a single perturbation."""
        flat = perturbation.flatten()
        magnitude = torch.norm(flat, p=2).item()
        
        if magnitude > 1e-10:
            direction = flat / magnitude
            self.directions.append(direction.cpu())
        
        self.perturbations.append(perturbation.cpu())
        self.magnitudes.append(magnitude)
    
    def compute_magnitude_diversity(self) -> float:
        """
        Compute diversity in perturbation magnitudes.
        
        Uses coefficient of variation (std/mean).
        
        Returns:
            Coefficient of variation
        """
        if len(self.magnitudes) < 2:
            return 0.0
        
        magnitudes = np.array(self.magnitudes)
        mean = np.mean(magnitudes)
        std = np.std(magnitudes)
        
        return std / mean if mean > 1e-10 else 0.0
    
    def compute_direction_diversity(self, sample_size: int = 100) -> float:
        """
        Compute diversity in perturbation directions.
        
        Uses average pairwise cosine distance.
        
        Args:
            sample_size: Max number of directions to sample for efficiency
        
        Returns:
            Average cosine distance in [0, 2]
        """
        if len(self.directions) < 2:
            return 0.0
        
        # Sample if too many
        directions = self.directions
        if len(directions) > sample_size:
            indices = np.random.choice(len(directions), sample_size, replace=False)
            directions = [directions[i] for i in indices]
        
        # Compute pairwise cosine similarities
        n = len(directions)
        total_distance = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = torch.dot(directions[i], directions[j]).item()
                cos_distance = 1 - cos_sim  # Convert to distance
                total_distance += cos_distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def compute_spatial_diversity(self) -> float:
        """
        Compute diversity in spatial distribution of perturbations.
        
        Measures how differently spread out perturbations are in space.
        
        Returns:
            Spatial diversity score
        """
        if len(self.perturbations) < 2:
            return 0.0
        
        # Compute spatial energy distribution for each perturbation
        distributions = []
        for pert in self.perturbations:
            # Sum across channels, normalize
            spatial_energy = (pert ** 2).sum(dim=0)  # [H, W]
            total = spatial_energy.sum()
            if total > 1e-10:
                normalized = (spatial_energy / total).flatten()
                distributions.append(normalized.numpy())
        
        if len(distributions) < 2:
            return 0.0
        
        # Compute average pairwise KL divergence
        n = len(distributions)
        total_div = 0.0
        count = 0
        
        for i in range(min(n, 50)):  # Sample for efficiency
            for j in range(i + 1, min(n, 50)):
                p = distributions[i] + 1e-10
                q = distributions[j] + 1e-10
                kl = np.sum(p * np.log(p / q))
                total_div += kl
                count += 1
        
        return total_div / count if count > 0 else 0.0
    
    def compute(self) -> DiversityResult:
        """Compute all perturbation diversity metrics."""
        return DiversityResult(
            name="PerturbationDiversity",
            value=self.compute_direction_diversity(),
            details={
                "magnitude_diversity": self.compute_magnitude_diversity(),
                "direction_diversity": self.compute_direction_diversity(),
                "spatial_diversity": self.compute_spatial_diversity(),
                "n_perturbations": len(self.perturbations)
            }
        )


class VulnerabilityCoverage:
    """
    Tracks which specific vulnerabilities are discovered.
    
    A vulnerability is defined as a (sample_idx, class_transition) pair,
    representing a specific misclassification that can be induced.
    """
    
    def __init__(self, n_classes: int):
        """
        Args:
            n_classes: Number of classes in the classification task
        """
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.vulnerabilities: Set[Tuple[int, int, int]] = set()
        self.class_transitions: Dict[Tuple[int, int], int] = defaultdict(int)
        self.per_sample_vulnerabilities: Dict[int, Set[int]] = defaultdict(set)
        self.total_samples = 0
    
    def update(
        self,
        sample_idx: int,
        true_label: int,
        adversarial_pred: int,
        original_correct: bool = True
    ):
        """
        Record a discovered vulnerability.
        
        Args:
            sample_idx: Index of the sample
            true_label: Ground truth label
            adversarial_pred: Prediction after attack
            original_correct: Whether original prediction was correct
        """
        self.total_samples = max(self.total_samples, sample_idx + 1)
        
        # Only count if original was correct and attack changed prediction
        if original_correct and adversarial_pred != true_label:
            # Record (sample, true_class, adversarial_class) triple
            vulnerability = (sample_idx, true_label, adversarial_pred)
            self.vulnerabilities.add(vulnerability)
            
            # Track class transition patterns
            transition = (true_label, adversarial_pred)
            self.class_transitions[transition] += 1
            
            # Track per-sample vulnerabilities
            self.per_sample_vulnerabilities[sample_idx].add(adversarial_pred)
    
    def update_batch(
        self,
        sample_indices: List[int],
        true_labels: torch.Tensor,
        adversarial_preds: torch.Tensor,
        original_preds: torch.Tensor
    ):
        """Update with a batch."""
        for i, (idx, true, adv, orig) in enumerate(zip(
            sample_indices,
            true_labels.cpu().numpy(),
            adversarial_preds.cpu().numpy(),
            original_preds.cpu().numpy()
        )):
            original_correct = (orig == true)
            self.update(idx, int(true), int(adv), original_correct)
    
    def compute_vulnerability_count(self) -> int:
        """Get total unique vulnerabilities discovered."""
        return len(self.vulnerabilities)
    
    def compute_sample_coverage(self) -> float:
        """
        Compute fraction of samples with at least one vulnerability.
        
        Returns:
            Coverage in [0, 1]
        """
        if self.total_samples == 0:
            return 0.0
        return len(self.per_sample_vulnerabilities) / self.total_samples
    
    def compute_transition_entropy(self) -> float:
        """
        Compute entropy of class transition distribution.
        
        Higher entropy means more diverse misclassifications.
        
        Returns:
            Entropy value
        """
        if not self.class_transitions:
            return 0.0
        
        total = sum(self.class_transitions.values())
        probs = np.array(list(self.class_transitions.values())) / total
        
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    def compute_avg_vulnerabilities_per_sample(self) -> float:
        """
        Average number of different misclassifications per sample.
        
        Returns:
            Average vulnerability count
        """
        if not self.per_sample_vulnerabilities:
            return 0.0
        
        counts = [len(v) for v in self.per_sample_vulnerabilities.values()]
        return float(np.mean(counts))
    
    def get_most_common_transitions(self, k: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get the k most common class transitions.
        
        Args:
            k: Number of transitions to return
        
        Returns:
            List of ((from_class, to_class), count) tuples
        """
        sorted_transitions = sorted(
            self.class_transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_transitions[:k]
    
    def compute(self) -> DiversityResult:
        """Compute all vulnerability coverage metrics."""
        return DiversityResult(
            name="VulnerabilityCoverage",
            value=self.compute_sample_coverage(),
            details={
                "vulnerability_count": self.compute_vulnerability_count(),
                "sample_coverage": self.compute_sample_coverage(),
                "transition_entropy": self.compute_transition_entropy(),
                "avg_vulnerabilities_per_sample": self.compute_avg_vulnerabilities_per_sample(),
                "most_common_transitions": self.get_most_common_transitions(5),
                "total_samples_seen": self.total_samples,
                "n_unique_transitions": len(self.class_transitions)
            }
        )


class ModelRegionCoverage:
    """
    Tracks which regions of the model's decision space are explored.
    
    Uses confidence scores and predictions to identify distinct
    behavioral regions of the model.
    """
    
    def __init__(self, n_classes: int, n_bins: int = 10):
        """
        Args:
            n_classes: Number of classes
            n_bins: Number of confidence bins
        """
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        # Track (predicted_class, confidence_bin) pairs
        self.regions_visited: Set[Tuple[int, int]] = set()
        self.region_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.boundary_crossings = 0
        self.total_attacks = 0
    
    def _get_confidence_bin(self, confidence: float) -> int:
        """Map confidence to bin index."""
        return min(int(confidence * self.n_bins), self.n_bins - 1)
    
    def update(
        self,
        original_pred: int,
        original_conf: float,
        adversarial_pred: int,
        adversarial_conf: float
    ):
        """
        Record a model query result.
        
        Args:
            original_pred: Original prediction
            original_conf: Original confidence
            adversarial_pred: Prediction after attack
            adversarial_conf: Confidence after attack
        """
        self.total_attacks += 1
        
        # Track region visited by adversarial example
        adv_conf_bin = self._get_confidence_bin(adversarial_conf)
        region = (adversarial_pred, adv_conf_bin)
        self.regions_visited.add(region)
        self.region_counts[region] += 1
        
        # Track boundary crossings
        if original_pred != adversarial_pred:
            self.boundary_crossings += 1
    
    def compute_region_coverage(self) -> float:
        """
        Compute fraction of (class, confidence) regions visited.
        
        Returns:
            Coverage in [0, 1]
        """
        total_regions = self.n_classes * self.n_bins
        return len(self.regions_visited) / total_regions
    
    def compute_boundary_rate(self) -> float:
        """
        Compute rate of decision boundary crossings.
        
        Returns:
            Rate in [0, 1]
        """
        if self.total_attacks == 0:
            return 0.0
        return self.boundary_crossings / self.total_attacks
    
    def compute_region_entropy(self) -> float:
        """
        Compute entropy of region visit distribution.
        
        Returns:
            Entropy value
        """
        if not self.region_counts:
            return 0.0
        
        total = sum(self.region_counts.values())
        probs = np.array(list(self.region_counts.values())) / total
        
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    def compute(self) -> DiversityResult:
        """Compute all region coverage metrics."""
        return DiversityResult(
            name="ModelRegionCoverage",
            value=self.compute_region_coverage(),
            details={
                "region_coverage": self.compute_region_coverage(),
                "boundary_rate": self.compute_boundary_rate(),
                "region_entropy": self.compute_region_entropy(),
                "n_regions_visited": len(self.regions_visited),
                "total_possible_regions": self.n_classes * self.n_bins,
                "total_attacks": self.total_attacks,
                "boundary_crossings": self.boundary_crossings
            }
        )


class ComprehensiveDiversityMetrics:
    """
    Unified interface for all diversity metrics.
    """
    
    def __init__(self, n_classes: int):
        """
        Args:
            n_classes: Number of classes in classification task
        """
        self.n_classes = n_classes
        self.strategy = StrategyDiversity()
        self.perturbation = PerturbationDiversity()
        self.vulnerability = VulnerabilityCoverage(n_classes)
        self.region = ModelRegionCoverage(n_classes)
    
    def reset(self):
        """Reset all metrics."""
        self.strategy.reset()
        self.perturbation.reset()
        self.vulnerability.reset()
        self.region.reset()
    
    def update(
        self,
        sample_idx: int,
        strategy: str,
        perturbation: torch.Tensor,
        true_label: int,
        original_pred: int,
        original_conf: float,
        adversarial_pred: int,
        adversarial_conf: float
    ):
        """
        Update all metrics with a single attack result.
        
        Args:
            sample_idx: Index of sample
            strategy: Attack strategy used
            perturbation: The perturbation tensor
            true_label: Ground truth label
            original_pred: Original prediction
            original_conf: Original confidence
            adversarial_pred: Adversarial prediction
            adversarial_conf: Adversarial confidence
        """
        success = (original_pred == true_label) and (adversarial_pred != true_label)
        
        self.strategy.update(strategy, success)
        self.perturbation.update(perturbation)
        self.vulnerability.update(
            sample_idx, true_label, adversarial_pred,
            original_correct=(original_pred == true_label)
        )
        self.region.update(original_pred, original_conf, adversarial_pred, adversarial_conf)
    
    def compute(self) -> Dict[str, DiversityResult]:
        """Compute all diversity metrics."""
        return {
            "strategy": self.strategy.compute(),
            "perturbation": self.perturbation.compute(),
            "vulnerability": self.vulnerability.compute(),
            "region": self.region.compute()
        }
    
    def summary(self) -> Dict[str, float]:
        """Get summary as simple dict."""
        results = self.compute()
        return {
            "strategy_diversity": results["strategy"].value,
            "strategy_coverage": results["strategy"].details["coverage"],
            "perturbation_diversity": results["perturbation"].value,
            "vulnerability_coverage": results["vulnerability"].value,
            "region_coverage": results["region"].value,
            "boundary_rate": results["region"].details["boundary_rate"]
        }
