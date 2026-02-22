"""
Unit tests for evaluation metrics.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    AttackSuccessRate,
    SSIM,
    PSNR,
    PerturbationMetrics,
    ComprehensiveMetrics,
    StrategyDiversity,
    VulnerabilityCoverage,
    QueryCounter,
    TimeTracker
)


class TestAttackSuccessRate(unittest.TestCase):
    """Tests for ASR metric."""
    
    def test_perfect_attack(self):
        asr = AttackSuccessRate(targeted=False)
        original_preds = torch.tensor([0, 1, 2, 3])
        adversarial_preds = torch.tensor([1, 2, 3, 0])
        true_labels = torch.tensor([0, 1, 2, 3])
        
        asr.update(original_preds, adversarial_preds, true_labels)
        result = asr.compute()
        
        self.assertEqual(result.value, 1.0)
    
    def test_no_attack_success(self):
        asr = AttackSuccessRate(targeted=False)
        original_preds = torch.tensor([0, 1, 2, 3])
        adversarial_preds = torch.tensor([0, 1, 2, 3])
        true_labels = torch.tensor([0, 1, 2, 3])
        
        asr.update(original_preds, adversarial_preds, true_labels)
        result = asr.compute()
        
        self.assertEqual(result.value, 0.0)
    
    def test_partial_success(self):
        asr = AttackSuccessRate(targeted=False)
        original_preds = torch.tensor([0, 1, 2, 3])
        adversarial_preds = torch.tensor([1, 1, 3, 3])
        true_labels = torch.tensor([0, 1, 2, 3])
        
        asr.update(original_preds, adversarial_preds, true_labels)
        result = asr.compute()
        
        self.assertEqual(result.value, 0.5)


class TestSSIM(unittest.TestCase):
    """Tests for SSIM metric."""
    
    def test_identical_images(self):
        ssim = SSIM()
        img = torch.rand(4, 3, 32, 32)
        ssim.update(img, img)
        result = ssim.compute()
        self.assertAlmostEqual(result.value, 1.0, places=5)
    
    def test_different_images(self):
        ssim = SSIM()
        img1 = torch.zeros(4, 3, 32, 32)
        img2 = torch.ones(4, 3, 32, 32)
        ssim.update(img1, img2)
        result = ssim.compute()
        self.assertLess(result.value, 0.5)
    
    def test_ssim_range(self):
        ssim = SSIM()
        img1 = torch.rand(4, 3, 32, 32)
        img2 = torch.rand(4, 3, 32, 32)
        ssim.update(img1, img2)
        result = ssim.compute()
        self.assertGreaterEqual(result.value, -1)
        self.assertLessEqual(result.value, 1)


class TestPSNR(unittest.TestCase):
    """Tests for PSNR metric."""
    
    def test_identical_images(self):
        psnr = PSNR()
        img = torch.rand(4, 3, 32, 32)
        psnr.update(img, img)
        result = psnr.compute()
        # PSNR should be very high for identical images
        self.assertGreater(result.value, 50)
    
    def test_small_perturbation(self):
        psnr = PSNR()
        img = torch.rand(4, 3, 32, 32)
        perturbed = img + torch.randn_like(img) * 0.01
        perturbed = torch.clamp(perturbed, 0, 1)
        psnr.update(img, perturbed)
        result = psnr.compute()
        # Small perturbation should give high PSNR
        self.assertGreater(result.value, 30)


class TestPerturbationMetrics(unittest.TestCase):
    """Tests for perturbation norm metrics."""
    
    def test_zero_perturbation(self):
        metrics = PerturbationMetrics()
        img = torch.rand(4, 3, 32, 32)
        metrics.update(img, img)
        results = metrics.compute()
        
        self.assertAlmostEqual(results['L_inf'].value, 0, places=5)
        self.assertAlmostEqual(results['L_2'].value, 0, places=5)
    
    def test_bounded_perturbation(self):
        metrics = PerturbationMetrics()
        img = torch.rand(4, 3, 32, 32)
        perturbed = img + 0.03  # Uniform perturbation
        perturbed = torch.clamp(perturbed, 0, 1)
        metrics.update(img, perturbed)
        results = metrics.compute()
        
        self.assertLessEqual(results['L_inf'].value, 0.03 + 1e-5)


class TestStrategyDiversity(unittest.TestCase):
    """Tests for strategy diversity metrics."""
    
    def test_single_strategy(self):
        div = StrategyDiversity()
        for _ in range(10):
            div.update('pixel', success=True)
        result = div.compute()
        self.assertEqual(result.details['coverage'], 0.25)
    
    def test_all_strategies(self):
        div = StrategyDiversity()
        for strategy in ['pixel', 'frequency', 'patch', 'semantic']:
            div.update(strategy, success=True)
        result = div.compute()
        self.assertEqual(result.details['coverage'], 1.0)
    
    def test_entropy_uniform(self):
        div = StrategyDiversity()
        for _ in range(25):
            for strategy in ['pixel', 'frequency', 'patch', 'semantic']:
                div.update(strategy, success=True)
        result = div.compute()
        # Uniform distribution should have high entropy
        self.assertGreater(result.details['normalized_entropy'], 0.9)


class TestVulnerabilityCoverage(unittest.TestCase):
    """Tests for vulnerability coverage metrics."""
    
    def test_single_vulnerability(self):
        cov = VulnerabilityCoverage(n_classes=10)
        cov.update(sample_idx=0, true_label=0, adversarial_pred=1, original_correct=True)
        result = cov.compute()
        self.assertEqual(result.details['vulnerability_count'], 1)
    
    def test_multiple_vulnerabilities(self):
        cov = VulnerabilityCoverage(n_classes=10)
        for i in range(10):
            cov.update(sample_idx=i, true_label=i, adversarial_pred=(i+1) % 10, original_correct=True)
        result = cov.compute()
        self.assertEqual(result.details['vulnerability_count'], 10)


class TestQueryCounter(unittest.TestCase):
    """Tests for query counting."""
    
    def test_count_queries(self):
        counter = QueryCounter()
        counter.start_attack()
        counter.count_forward(10)
        counter.count_gradient(5)
        counter.end_attack()
        
        self.assertEqual(counter.total_queries, 15)
        self.assertEqual(counter.forward_queries, 10)
        self.assertEqual(counter.gradient_queries, 5)


class TestTimeTracker(unittest.TestCase):
    """Tests for time tracking."""
    
    def test_track_phase(self):
        tracker = TimeTracker()
        import time
        
        with tracker.track_phase('test'):
            time.sleep(0.1)
        
        self.assertGreater(tracker.phase_times['test'], 0.05)


if __name__ == '__main__':
    unittest.main()
