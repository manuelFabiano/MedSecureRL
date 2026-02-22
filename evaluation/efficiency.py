"""
Efficiency metrics for adversarial attack evaluation.

Measures computational costs: query counts, time, memory usage,
and attack efficiency ratios.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict


@dataclass
class EfficiencyResult:
    """Container for efficiency metric results."""
    name: str
    value: float
    unit: str
    details: Dict[str, Any] = field(default_factory=dict)


class QueryCounter:
    """
    Tracks model queries during adversarial attacks.
    
    Counts both forward passes and gradient computations.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.forward_queries = 0
        self.gradient_queries = 0
        self.per_attack_forward: List[int] = []
        self.per_attack_gradient: List[int] = []
        self._attack_forward = 0
        self._attack_gradient = 0
    
    def count_forward(self, batch_size: int = 1):
        """Record forward pass queries."""
        self.forward_queries += batch_size
        self._attack_forward += batch_size
    
    def count_gradient(self, batch_size: int = 1):
        """Record gradient computation queries."""
        self.gradient_queries += batch_size
        self._attack_gradient += batch_size
    
    def start_attack(self):
        """Start counting for a new attack."""
        self._attack_forward = 0
        self._attack_gradient = 0
    
    def end_attack(self):
        """End counting for current attack."""
        self.per_attack_forward.append(self._attack_forward)
        self.per_attack_gradient.append(self._attack_gradient)
    
    @property
    def total_queries(self) -> int:
        """Total queries (forward + gradient)."""
        return self.forward_queries + self.gradient_queries
    
    def compute_stats(self) -> Dict[str, float]:
        """Compute query statistics."""
        stats = {
            "total_queries": self.total_queries,
            "forward_queries": self.forward_queries,
            "gradient_queries": self.gradient_queries,
        }
        
        if self.per_attack_forward:
            per_attack_total = np.array(self.per_attack_forward) + np.array(self.per_attack_gradient)
            stats["mean_queries_per_attack"] = float(np.mean(per_attack_total))
            stats["std_queries_per_attack"] = float(np.std(per_attack_total))
            stats["min_queries_per_attack"] = float(np.min(per_attack_total))
            stats["max_queries_per_attack"] = float(np.max(per_attack_total))
            stats["n_attacks"] = len(self.per_attack_forward)
        
        return stats
    
    def compute(self) -> EfficiencyResult:
        """Compute query efficiency metrics."""
        stats = self.compute_stats()
        mean_queries = stats.get("mean_queries_per_attack", self.total_queries)
        
        return EfficiencyResult(
            name="QueryEfficiency",
            value=mean_queries,
            unit="queries/attack",
            details=stats
        )


class TimeTracker:
    """
    Tracks time spent on different phases of attack.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timers."""
        self.phase_times: Dict[str, float] = defaultdict(float)
        self.attack_times: List[float] = []
        self._current_phase: Optional[str] = None
        self._phase_start: Optional[float] = None
        self._attack_start: Optional[float] = None
    
    @contextmanager
    def track_phase(self, phase_name: str):
        """Context manager for timing a phase."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.phase_times[phase_name] += elapsed
    
    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        self._current_phase = phase_name
        self._phase_start = time.time()
    
    def end_phase(self):
        """End timing current phase."""
        if self._current_phase is not None and self._phase_start is not None:
            elapsed = time.time() - self._phase_start
            self.phase_times[self._current_phase] += elapsed
            self._current_phase = None
            self._phase_start = None
    
    def start_attack(self):
        """Start timing an attack."""
        self._attack_start = time.time()
    
    def end_attack(self):
        """End timing current attack."""
        if self._attack_start is not None:
            elapsed = time.time() - self._attack_start
            self.attack_times.append(elapsed)
            self._attack_start = None
    
    @property
    def total_time(self) -> float:
        """Total time across all phases."""
        return sum(self.phase_times.values())
    
    def compute_stats(self) -> Dict[str, float]:
        """Compute timing statistics."""
        stats = {
            "total_time": self.total_time,
            "phase_times": dict(self.phase_times),
        }
        
        if self.attack_times:
            stats["mean_time_per_attack"] = float(np.mean(self.attack_times))
            stats["std_time_per_attack"] = float(np.std(self.attack_times))
            stats["min_time_per_attack"] = float(np.min(self.attack_times))
            stats["max_time_per_attack"] = float(np.max(self.attack_times))
            stats["n_attacks"] = len(self.attack_times)
            stats["attacks_per_second"] = len(self.attack_times) / self.total_time if self.total_time > 0 else 0
        
        return stats
    
    def compute(self) -> EfficiencyResult:
        """Compute time efficiency metrics."""
        stats = self.compute_stats()
        mean_time = stats.get("mean_time_per_attack", self.total_time)
        
        return EfficiencyResult(
            name="TimeEfficiency",
            value=mean_time,
            unit="seconds/attack",
            details=stats
        )


class MemoryTracker:
    """
    Tracks GPU/CPU memory usage during attacks.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()
    
    def reset(self):
        """Reset memory tracking."""
        self.peak_memory: List[float] = []
        self.memory_samples: List[float] = []
        self._baseline_memory: Optional[float] = None
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            # CPU memory tracking is less precise
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        else:
            return self.get_current_memory()
    
    def set_baseline(self):
        """Set baseline memory before attacks."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._baseline_memory = self.get_current_memory()
    
    def sample_memory(self):
        """Take a memory sample."""
        self.memory_samples.append(self.get_current_memory())
    
    def record_peak(self):
        """Record peak memory for current attack."""
        peak = self.get_peak_memory()
        if self._baseline_memory is not None:
            peak -= self._baseline_memory
        self.peak_memory.append(max(0, peak))
        
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def compute_stats(self) -> Dict[str, float]:
        """Compute memory statistics."""
        stats = {
            "device": str(self.device),
            "baseline_memory_mb": self._baseline_memory or 0,
        }
        
        if self.peak_memory:
            stats["mean_peak_memory_mb"] = float(np.mean(self.peak_memory))
            stats["max_peak_memory_mb"] = float(np.max(self.peak_memory))
            stats["n_samples"] = len(self.peak_memory)
        
        if self.memory_samples:
            stats["mean_memory_mb"] = float(np.mean(self.memory_samples))
            stats["std_memory_mb"] = float(np.std(self.memory_samples))
        
        return stats
    
    def compute(self) -> EfficiencyResult:
        """Compute memory efficiency metrics."""
        stats = self.compute_stats()
        mean_peak = stats.get("mean_peak_memory_mb", 0)
        
        return EfficiencyResult(
            name="MemoryEfficiency",
            value=mean_peak,
            unit="MB",
            details=stats
        )


class AttackEfficiency:
    """
    Computes efficiency ratios combining success and cost.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated data."""
        self.successes: List[bool] = []
        self.queries: List[int] = []
        self.times: List[float] = []
        self.perturbation_norms: List[float] = []
    
    def update(
        self,
        success: bool,
        queries: int,
        time_seconds: float,
        perturbation_norm: float
    ):
        """
        Record an attack result.
        
        Args:
            success: Whether attack succeeded
            queries: Number of queries used
            time_seconds: Time taken
            perturbation_norm: L2 norm of perturbation
        """
        self.successes.append(success)
        self.queries.append(queries)
        self.times.append(time_seconds)
        self.perturbation_norms.append(perturbation_norm)
    
    def compute_success_per_query(self) -> float:
        """
        Compute success rate per query.
        
        Returns:
            Success rate normalized by query count
        """
        if not self.successes or sum(self.queries) == 0:
            return 0.0
        
        # Weighted success rate by query efficiency
        successes = np.array(self.successes).astype(float)
        queries = np.maximum(np.array(self.queries), 1)  # Clamp negatives
        
        # Success per query (higher is better)
        return float(successes.sum() / queries.sum())
    
    def compute_efficiency_score(self) -> float:
        """
        Compute combined efficiency score.
        
        Higher score = more success with less cost.
        
        Returns:
            Efficiency score
        """
        if not self.successes:
            return 0.0
        
        successes = np.array(self.successes).astype(float)
        queries = np.maximum(np.array(self.queries), 0) + 1  # Clamp negatives, avoid div by zero
        times = np.array(self.times) + 0.001
        norms = np.array(self.perturbation_norms) + 0.001
        
        # Per-attack efficiency: success / (queries * time * norm)
        per_attack_efficiency = successes / (np.sqrt(queries) * norms)
        
        return float(np.mean(per_attack_efficiency))
    
    def compute_marginal_success(self, query_budget: int = 100) -> float:
        """
        Compute success rate within a query budget.
        
        Args:
            query_budget: Maximum queries to consider
        
        Returns:
            Success rate for attacks within budget
        """
        if not self.successes:
            return 0.0
        
        within_budget = [s for s, q in zip(self.successes, self.queries) if q <= query_budget]
        
        if not within_budget:
            return 0.0
        
        return float(np.mean(within_budget))
    
    def compute(self) -> EfficiencyResult:
        """Compute attack efficiency metrics."""
        n_attacks = len(self.successes)
        n_successful = sum(self.successes)
        
        return EfficiencyResult(
            name="AttackEfficiency",
            value=self.compute_efficiency_score(),
            unit="score",
            details={
                "n_attacks": n_attacks,
                "n_successful": n_successful,
                "success_rate": n_successful / n_attacks if n_attacks > 0 else 0,
                "success_per_query": self.compute_success_per_query(),
                "efficiency_score": self.compute_efficiency_score(),
                "marginal_success_100q": self.compute_marginal_success(100),
                "marginal_success_50q": self.compute_marginal_success(50),
                "mean_queries": float(np.mean(self.queries)) if self.queries else 0,
                "mean_time": float(np.mean(self.times)) if self.times else 0,
                "mean_perturbation_norm": float(np.mean(self.perturbation_norms)) if self.perturbation_norms else 0
            }
        )


class ComprehensiveEfficiencyMetrics:
    """
    Unified interface for all efficiency metrics.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Device for memory tracking
        """
        self.queries = QueryCounter()
        self.time = TimeTracker()
        self.memory = MemoryTracker(device)
        self.efficiency = AttackEfficiency()
    
    def reset(self):
        """Reset all metrics."""
        self.queries.reset()
        self.time.reset()
        self.memory.reset()
        self.efficiency.reset()
    
    def start_attack(self):
        """Start tracking a new attack."""
        self.queries.start_attack()
        self.time.start_attack()
        self.memory.set_baseline()
    
    def end_attack(self, success: bool, perturbation_norm: float):
        """
        End tracking current attack.
        
        Args:
            success: Whether attack succeeded
            perturbation_norm: L2 norm of final perturbation
        """
        self.queries.end_attack()
        self.time.end_attack()
        self.memory.record_peak()
        
        # Get last recorded values
        queries = self.queries.per_attack_forward[-1] + self.queries.per_attack_gradient[-1]
        time_taken = self.time.attack_times[-1]
        
        self.efficiency.update(success, queries, time_taken, perturbation_norm)
    
    def count_forward(self, batch_size: int = 1):
        """Record forward pass."""
        self.queries.count_forward(batch_size)
    
    def count_gradient(self, batch_size: int = 1):
        """Record gradient computation."""
        self.queries.count_gradient(batch_size)
    
    @contextmanager
    def track_phase(self, phase_name: str):
        """Context manager for timing a phase."""
        with self.time.track_phase(phase_name):
            yield
    
    def compute(self) -> Dict[str, EfficiencyResult]:
        """Compute all efficiency metrics."""
        return {
            "queries": self.queries.compute(),
            "time": self.time.compute(),
            "memory": self.memory.compute(),
            "efficiency": self.efficiency.compute()
        }
    
    def summary(self) -> Dict[str, float]:
        """Get summary as simple dict."""
        results = self.compute()
        return {
            "total_queries": results["queries"].details.get("total_queries", 0),
            "mean_queries_per_attack": results["queries"].value,
            "mean_time_per_attack": results["time"].value,
            "mean_peak_memory_mb": results["memory"].value,
            "efficiency_score": results["efficiency"].value,
            "success_per_query": results["efficiency"].details.get("success_per_query", 0)
        }


def compute_pareto_efficiency(
    results: List[Dict[str, float]],
    objectives: List[str] = ["asr", "queries", "perturbation_norm"],
    maximize: List[bool] = [True, False, False]
) -> List[int]:
    """
    Find Pareto-efficient solutions from a set of attack results.
    
    Args:
        results: List of result dictionaries
        objectives: Keys to use as objectives
        maximize: Whether to maximize each objective
    
    Returns:
        Indices of Pareto-efficient solutions
    """
    n = len(results)
    if n == 0:
        return []
    
    # Extract objective values
    values = np.zeros((n, len(objectives)))
    for i, r in enumerate(results):
        for j, obj in enumerate(objectives):
            values[i, j] = r.get(obj, 0)
            # Negate if minimizing (we'll find maximal points)
            if not maximize[j]:
                values[i, j] = -values[i, j]
    
    # Find Pareto frontier
    is_efficient = np.ones(n, dtype=bool)
    
    for i in range(n):
        if is_efficient[i]:
            # Check if any other point dominates this one
            for j in range(n):
                if i != j and is_efficient[j]:
                    # j dominates i if j >= i on all objectives and j > i on at least one
                    dominates = (values[j] >= values[i]).all() and (values[j] > values[i]).any()
                    if dominates:
                        is_efficient[i] = False
                        break
    
    return list(np.where(is_efficient)[0])


def compute_speedup(
    baseline_time: float,
    method_time: float
) -> float:
    """
    Compute speedup ratio.
    
    Args:
        baseline_time: Time for baseline method
        method_time: Time for new method
    
    Returns:
        Speedup ratio (>1 means faster)
    """
    if method_time <= 0:
        return float('inf')
    return baseline_time / method_time