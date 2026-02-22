"""
Unit tests for attack implementations.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from attacks import PixelAttack, FrequencyAttack, PatchAttack, SemanticAttack
from attacks.baselines import FGSM, PGD, CarliniWagner


class MockModel(torch.nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(16, n_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class TestPixelAttack(unittest.TestCase):
    """Tests for pixel-based attacks."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()
        self.images = torch.rand(4, 3, 28, 28, device=self.device)
        self.labels = torch.randint(0, 10, (4,), device=self.device)
    
    def test_fgsm_output_shape(self):
        attack = FGSM(self.model, epsilon=0.03, device=self.device)
        result = attack.attack(self.images, self.labels)
        self.assertEqual(result.adversarial.shape, self.images.shape)
    
    def test_fgsm_perturbation_bounded(self):
        attack = FGSM(self.model, epsilon=0.03, device=self.device)
        result = attack.attack(self.images, self.labels)
        perturbation = (result.adversarial - self.images).abs()
        self.assertTrue((perturbation <= 0.03 + 1e-6).all())
    
    def test_pgd_output_shape(self):
        attack = PGD(self.model, epsilon=0.03, iterations=10, device=self.device)
        result = attack.attack(self.images, self.labels)
        self.assertEqual(result.adversarial.shape, self.images.shape)
    
    def test_pgd_perturbation_bounded(self):
        attack = PGD(self.model, epsilon=0.03, iterations=10, device=self.device)
        result = attack.attack(self.images, self.labels)
        perturbation = (result.adversarial - self.images).abs()
        self.assertTrue((perturbation <= 0.03 + 1e-6).all())
    
    def test_adversarial_in_valid_range(self):
        attack = FGSM(self.model, epsilon=0.1, device=self.device)
        result = attack.attack(self.images, self.labels)
        self.assertTrue((result.adversarial >= 0).all())
        self.assertTrue((result.adversarial <= 1).all())


class TestFrequencyAttack(unittest.TestCase):
    """Tests for frequency-domain attacks."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()
        self.images = torch.rand(4, 3, 28, 28, device=self.device)
        self.labels = torch.randint(0, 10, (4,), device=self.device)
    
    def test_frequency_attack_output_shape(self):
        attack = FrequencyAttack(
            model=self.model, epsilon=0.03,
            band='low', device=self.device
        )
        result = attack.attack(self.images, self.labels)
        self.assertEqual(result.adversarial.shape, self.images.shape)
    
    def test_different_bands(self):
        for band in ['low', 'mid', 'high']:
            attack = FrequencyAttack(
                model=self.model, epsilon=0.03,
                band=band, device=self.device
            )
            result = attack.attack(self.images, self.labels)
            self.assertEqual(result.adversarial.shape, self.images.shape)


class TestPatchAttack(unittest.TestCase):
    """Tests for patch-based attacks."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()
        self.images = torch.rand(4, 3, 28, 28, device=self.device)
        self.labels = torch.randint(0, 10, (4,), device=self.device)
    
    def test_patch_attack_output_shape(self):
        attack = PatchAttack(
            model=self.model, patch_size=7,
            device=self.device
        )
        result = attack.attack(self.images, self.labels)
        self.assertEqual(result.adversarial.shape, self.images.shape)
    
    def test_patch_localized(self):
        attack = PatchAttack(
            model=self.model, patch_size=7,
            device=self.device
        )
        result = attack.attack(self.images, self.labels)
        diff = (result.adversarial - self.images).abs()
        # Most pixels should be unchanged
        unchanged_ratio = (diff < 1e-6).float().mean()
        self.assertGreater(unchanged_ratio, 0.5)


class TestSemanticAttack(unittest.TestCase):
    """Tests for semantic attacks."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()
        self.images = torch.rand(4, 3, 28, 28, device=self.device)
        self.labels = torch.randint(0, 10, (4,), device=self.device)
    
    def test_semantic_attack_output_shape(self):
        attack = SemanticAttack(
            model=self.model, device=self.device
        )
        result = attack.attack(self.images, self.labels)
        self.assertEqual(result.adversarial.shape, self.images.shape)
    
    def test_adversarial_in_valid_range(self):
        attack = SemanticAttack(model=self.model, device=self.device)
        result = attack.attack(self.images, self.labels)
        self.assertTrue((result.adversarial >= 0).all())
        self.assertTrue((result.adversarial <= 1).all())


if __name__ == '__main__':
    unittest.main()
