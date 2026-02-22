"""
Evaluation module for MedSecure adversarial robustness testing.

This module provides comprehensive evaluation capabilities:
- Metrics: ASR, SSIM, PSNR, LPIPS for attack quality assessment
- Diversity: Coverage and uniqueness of discovered vulnerabilities
- Efficiency: Query count and computational cost metrics
- Visualization: Plots, heatmaps, and reports
"""

from .metrics import (
    MetricResult,
    AttackSuccessRate,
    SSIM,
    PSNR,
    LPIPS_Metric,
    PerturbationMetrics,
    ComprehensiveMetrics,
    compute_fooling_rate,
    compute_transfer_rate
)
from .diversity import (
    DiversityResult,
    StrategyDiversity,
    PerturbationDiversity,
    VulnerabilityCoverage,
    ModelRegionCoverage,
    ComprehensiveDiversityMetrics
)
from .efficiency import (
    EfficiencyResult,
    QueryCounter,
    TimeTracker,
    MemoryTracker,
    AttackEfficiency,
    ComprehensiveEfficiencyMetrics,
    compute_pareto_efficiency,
    compute_speedup
)
from .visualization import (
    MetricsPlotter,
    PerturbationVisualizer,
    ConfusionMatrixPlotter,
    TrainingVisualizer,
    ReportGenerator,
    create_summary_figure
)

__all__ = [
    # Metrics
    "MetricResult",
    "AttackSuccessRate",
    "SSIM",
    "PSNR",
    "LPIPS_Metric",
    "PerturbationMetrics",
    "ComprehensiveMetrics",
    "compute_fooling_rate",
    "compute_transfer_rate",
    # Diversity
    "DiversityResult",
    "StrategyDiversity",
    "PerturbationDiversity",
    "VulnerabilityCoverage",
    "ModelRegionCoverage",
    "ComprehensiveDiversityMetrics",
    # Efficiency
    "EfficiencyResult",
    "QueryCounter",
    "TimeTracker",
    "MemoryTracker",
    "AttackEfficiency",
    "ComprehensiveEfficiencyMetrics",
    "compute_pareto_efficiency",
    "compute_speedup",
    # Visualization
    "MetricsPlotter",
    "PerturbationVisualizer",
    "ConfusionMatrixPlotter",
    "TrainingVisualizer",
    "ReportGenerator",
    "create_summary_figure"
]
