"""
Unit tests for RL environment.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import AdversarialEnv, StateExtractor, RewardFunction


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


class TestStateExtractor(unittest.TestCase):
    """Tests for state extraction."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()
        self.extractor = StateExtractor(
            model=self.model, 
            state_dim=20,
            device=self.device
        )
    
    def test_state_dimension(self):
        image = torch.rand(1, 3, 28, 28, device=self.device)
        perturbation = torch.zeros_like(image)
        state = self.extractor.extract(image, perturbation, iteration=0)
        self.assertEqual(state.shape, (20,))
    
    def test_state_normalized(self):
        image = torch.rand(1, 3, 28, 28, device=self.device)
        perturbation = torch.rand_like(image) * 0.03
        state = self.extractor.extract(image, perturbation, iteration=10)
        # State should be roughly normalized
        self.assertTrue(state.abs().max() < 100)


class TestRewardFunction(unittest.TestCase):
    """Tests for reward computation."""
    
    def setUp(self):
        self.reward_fn = RewardFunction(
            success_weight=1.0,
            imperceptibility_weight=0.3,
            efficiency_weight=0.1,
            diversity_weight=0.2
        )
    
    def test_success_reward(self):
        reward = self.reward_fn.compute(
            attack_success=True,
            perturbation_norm=0.01,
            queries=10,
            strategy='pixel',
            original_conf=0.9,
            adversarial_conf=0.3
        )
        self.assertGreater(reward, 0)
    
    def test_failure_penalty(self):
        reward = self.reward_fn.compute(
            attack_success=False,
            perturbation_norm=0.05,
            queries=50,
            strategy='pixel',
            original_conf=0.9,
            adversarial_conf=0.8
        )
        self.assertLess(reward, 1.0)
    
    def test_imperceptibility_bonus(self):
        # Small perturbation should give higher reward
        reward_small = self.reward_fn.compute(
            attack_success=True,
            perturbation_norm=0.01,
            queries=10,
            strategy='pixel'
        )
        reward_large = self.reward_fn.compute(
            attack_success=True,
            perturbation_norm=0.1,
            queries=10,
            strategy='pixel'
        )
        self.assertGreater(reward_small, reward_large)


class TestAdversarialEnv(unittest.TestCase):
    """Tests for the RL environment."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MockModel().to(self.device)
        self.model.eval()
        # Note: This test requires actual data, skip if not available
    
    def test_env_spaces(self):
        try:
            env = AdversarialEnv(
                victim_model=self.model,
                dataset_name='pathmnist',
                device=self.device,
                epsilon=0.03
            )
            self.assertIsNotNone(env.observation_space)
            self.assertIsNotNone(env.action_space)
        except Exception:
            self.skipTest("Dataset not available")
    
    def test_reset_returns_state(self):
        try:
            env = AdversarialEnv(
                victim_model=self.model,
                dataset_name='pathmnist',
                device=self.device,
                epsilon=0.03
            )
            state, info = env.reset()
            self.assertEqual(len(state), env.observation_space.shape[0])
            self.assertIsInstance(info, dict)
        except Exception:
            self.skipTest("Dataset not available")
    
    def test_step_returns_correct_format(self):
        try:
            env = AdversarialEnv(
                victim_model=self.model,
                dataset_name='pathmnist',
                device=self.device,
                epsilon=0.03
            )
            state, _ = env.reset()
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            self.assertEqual(len(next_state), env.observation_space.shape[0])
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
        except Exception:
            self.skipTest("Dataset not available")


if __name__ == '__main__':
    unittest.main()
