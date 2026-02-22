"""
Multi-objective Reward Function for Adversarial Attack Generation.

This module implements a sophisticated reward function that balances:
1. Attack success rate
2. Perturbation imperceptibility  
3. Query efficiency
4. Attack diversity

The reward function is designed to encourage the RL agent to find
diverse, efficient, and imperceptible adversarial attacks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


@dataclass
class RewardComponents:
    """Container for individual reward components."""
    success: float = 0.0
    imperceptibility: float = 0.0
    efficiency: float = 0.0
    diversity: float = 0.0
    confidence_reduction: float = 0.0
    curriculum_bonus: float = 0.0
    epsilon_efficiency: float = 0.0  # Bonus for using low epsilon
    total: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "success": self.success,
            "imperceptibility": self.imperceptibility,
            "efficiency": self.efficiency,
            "diversity": self.diversity,
            "confidence_reduction": self.confidence_reduction,
            "curriculum_bonus": self.curriculum_bonus,
            "epsilon_efficiency": self.epsilon_efficiency,
            "total": self.total
        }


class RewardFunction:
    """
    Multi-objective reward function for adversarial attack generation.
    
    The reward combines multiple objectives with configurable weights:
    - Success: Binary reward for successful misclassification
    - Imperceptibility: Reward for small, perceptually invisible perturbations
    - Efficiency: Reward for using fewer queries
    - Diversity: Reward for exploring different attack strategies
    
    Reward Shaping:
    - Progressive rewards for reducing model confidence
    - Curriculum-based bonuses that change over training
    - Exploration bonuses for trying new strategies
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None,
        mode: str = "standard"
    ):
        """
        Initialize the reward function.
        
        Args:
            config: Reward configuration dictionary
            device: Computation device
            normalize_stats: Dict with 'mean' and 'std' for image denormalization
            mode: Reward mode - "standard" for normal victims, "robust" for adversarially trained
        """
        self.device = device
        self.config = config
        self.normalize_stats = normalize_stats
        self.mode = mode
        
        # Mode-specific settings
        if mode == "robust":
            # Relaxed settings for attacking robust models
            self.ssim_threshold = 0.80  # Lower threshold
            default_w_epsilon_efficiency = 0.1  # Less penalty for high epsilon
            default_w_imperceptibility = 0.1  # Less focus on imperceptibility
            print(f"[RewardFunction] Using ROBUST mode: SSIM threshold={self.ssim_threshold}")
        else:
            # Standard settings
            self.ssim_threshold = 0.90
            default_w_epsilon_efficiency = 0.5
            default_w_imperceptibility = 0.3
        
        # Reward weights
        reward_config = config.get("reward", {})
        weights = reward_config.get("weights", {})
        
        self.w_success = weights.get("success", 1.0)
        self.w_imperceptibility = weights.get("imperceptibility", default_w_imperceptibility)
        self.w_efficiency = weights.get("efficiency", 0.05)
        self.w_diversity = weights.get("diversity", 0.02)
        self.w_epsilon_efficiency = weights.get("epsilon_efficiency", default_w_epsilon_efficiency)
        
        # Thresholds and bounds
        self.success_reward = reward_config.get("success_reward", 1.0)
        self.failure_penalty = reward_config.get("failure_penalty", -0.5)
        self.max_queries = reward_config.get("max_queries", 1000)
        self.epsilon_budget = reward_config.get("epsilon_budget", 0.3)
        
        # LPIPS for perceptual similarity (optional)
        self.lpips_model = None
        if LPIPS_AVAILABLE and reward_config.get("use_lpips", True):
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
                self.lpips_model.eval()
            except Exception as e:
                print(f"Warning: Could not load LPIPS model: {e}")
        
        # Diversity tracking
        self.strategy_counts = defaultdict(int)
        self.strategy_successes = defaultdict(int)
        self.recent_strategies = []
        self.diversity_window = reward_config.get("diversity_window", 100)
        
        # Curriculum state
        self.curriculum_phase = 0
        self.training_step = 0
        
    def compute_reward(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        labels: torch.Tensor,
        model_output: torch.Tensor,
        attack_info: Dict[str, Any],
        curriculum_phase: int = 0
    ) -> RewardComponents:
        """
        Compute multi-objective reward.
        
        SSIM reward is continuous and centered on 0.90:
        - SSIM < 0.90: penalty that decreases as SSIM approaches 0.90
        - SSIM >= 0.90: reward that increases with SSIM
        
        This gives smooth gradients for the agent to learn optimal epsilon.
        """
        self.training_step += 1
        self.curriculum_phase = curriculum_phase
        
        components = RewardComponents()
        
        # Compute quality metrics
        ssim_val = self._compute_ssim(original, adversarial)
        psnr_val = self._compute_psnr(original, adversarial)
        
        # Check if attack caused misclassification
        predictions = model_output.argmax(dim=-1)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        misclassified = (predictions != labels).float().mean().item() > 0.5
        
        # Store in attack_info
        attack_info["ssim"] = ssim_val
        attack_info["psnr"] = psnr_val
        attack_info["misclassified"] = misclassified
        
        # =================================================================
        # SUCCESS REWARD - ONLY IF QUALITY SUCCESS (misclass + SSIM >= threshold)
        # =================================================================
        ssim_threshold = self.ssim_threshold  # Mode-dependent
        quality_success = misclassified and ssim_val >= ssim_threshold
        
        if quality_success:
            # True success: misclassified with good quality
            components.success = self.success_reward
            
            # SSIM bonus for exceeding threshold (scaled by mode)
            ssim_bonus = (ssim_val - ssim_threshold) / (1.0 - ssim_threshold)  # 0 to 1
            
            if self.mode == "robust":
                # For robust mode, bonus for SSIM >= 0.90 (good quality despite hard target)
                if ssim_val >= 0.90:
                    ssim_bonus += 0.5
                elif ssim_val >= 0.85:
                    ssim_bonus += 0.25
            else:
                # Standard mode
                if ssim_val >= 0.98:
                    ssim_bonus += 0.5
                elif ssim_val >= 0.95:
                    ssim_bonus += 0.25
            
            # PSNR bonus (relaxed for robust mode)
            psnr_threshold_high = 35 if self.mode == "robust" else 40
            psnr_threshold_low = 30 if self.mode == "robust" else 35
            
            if psnr_val >= psnr_threshold_high:
                psnr_bonus = 0.3
            elif psnr_val >= psnr_threshold_low:
                psnr_bonus = 0.1
            else:
                psnr_bonus = 0.0
            
            components.imperceptibility = ssim_bonus + psnr_bonus
            
            # =================================================================
            # EPSILON EFFICIENCY BONUS - reward for using lower epsilon
            # =================================================================
            epsilon_used = attack_info.get("epsilon", self.epsilon_budget)
            epsilon_min = attack_info.get("epsilon_min", self.epsilon_budget * 0.2)
            epsilon_max = attack_info.get("epsilon_max", self.epsilon_budget)
            
            if epsilon_max > epsilon_min:
                # Bonus = 1 when epsilon = epsilon_min, 0 when epsilon = epsilon_max
                epsilon_ratio = (epsilon_max - epsilon_used) / (epsilon_max - epsilon_min)
                epsilon_ratio = max(0.0, min(1.0, epsilon_ratio))  # Clip to [0, 1]
                components.epsilon_efficiency = epsilon_ratio
            else:
                components.epsilon_efficiency = 0.0
            
        elif misclassified:
            # Misclassified but SSIM too low - this is a FAILURE
            # Penalty proportional to how far below threshold
            distance = ssim_threshold - ssim_val
            components.success = self.failure_penalty - distance * 2  # Extra penalty
            components.imperceptibility = 0.0
            
        else:
            # No misclassification - regular failure
            components.success = self.failure_penalty
            components.imperceptibility = 0.0
        # Store in attack_info for env
        attack_info["success"] = quality_success
        attack_info["quality_ok"] = ssim_val >= ssim_threshold
        
        # Efficiency reward - only if quality success
        if quality_success:
            components.efficiency = self._compute_efficiency_reward(attack_info)
        else:
            components.efficiency = 0.0
        
        # Diversity reward
        components.diversity = self._compute_diversity_reward(attack_info)
        
        # Confidence reduction for failed attacks
        if not misclassified:
            components.confidence_reduction = self._compute_confidence_reduction(
                model_output, labels, attack_info
            )
        else:
            components.confidence_reduction = 0.0
        
        # Curriculum bonus
        components.curriculum_bonus = self._compute_curriculum_bonus(
            components, curriculum_phase
        )
        
        # Weighted total
        components.total = (
            self.w_success * components.success +
            self.w_imperceptibility * components.imperceptibility +
            self.w_efficiency * components.efficiency +
            self.w_diversity * components.diversity +
            self.w_epsilon_efficiency * components.epsilon_efficiency +
            components.confidence_reduction +
            components.curriculum_bonus
        )
        
        return components
    
    def _compute_ssim(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        """Compute SSIM between original and adversarial images."""
        # Denormalize if necessary
        if hasattr(self, 'normalize_stats') and self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats.get('mean', [0.5])).view(1, -1, 1, 1).to(original.device)
            std = torch.tensor(self.normalize_stats.get('std', [0.5])).view(1, -1, 1, 1).to(original.device)
            orig_denorm = original * std + mean
            adv_denorm = adversarial * std + mean
        else:
            orig_denorm = original
            adv_denorm = adversarial
        
        try:
            from skimage.metrics import structural_similarity
            orig_np = orig_denorm.squeeze().cpu().detach().numpy().clip(0, 1)
            adv_np = adv_denorm.squeeze().cpu().detach().numpy().clip(0, 1)
            
            if orig_np.ndim == 3:
                orig_np = orig_np.transpose(1, 2, 0)
                adv_np = adv_np.transpose(1, 2, 0)
            
            return structural_similarity(
                orig_np, adv_np, 
                data_range=1.0,
                channel_axis=-1 if orig_np.ndim == 3 else None
            )
        except:
            return 0.5
    
    def _compute_psnr(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        """Compute PSNR between original and adversarial images."""
        if hasattr(self, 'normalize_stats') and self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats.get('mean', [0.5])).view(1, -1, 1, 1).to(original.device)
            std = torch.tensor(self.normalize_stats.get('std', [0.5])).view(1, -1, 1, 1).to(original.device)
            orig_denorm = original * std + mean
            adv_denorm = adversarial * std + mean
        else:
            orig_denorm = original
            adv_denorm = adversarial
        
        try:
            mse = ((orig_denorm - adv_denorm) ** 2).mean().item()
            if mse > 1e-10:
                return 10 * np.log10(1.0 / mse)
            return 100.0
        except:
            return 30.0
    
    def _compute_success_reward(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        attack_info: Dict[str, Any]
    ) -> float:
        """Compute reward for attack success.
        
        Primary objective: successful misclassification.
        Secondary: bonus for efficiency (lower epsilon while still succeeding).
        """
        predictions = model_output.argmax(dim=-1)
        
        # Handle both single and batched inputs
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
            
        success = (predictions != labels).float().mean().item()
        
        if success > 0.5:  # Majority successful
            # Base success reward
            base_reward = self.success_reward
            
            # Small bonus for decisive success (high confidence in wrong class)
            confidence = F.softmax(model_output, dim=-1)
            wrong_confidence = 1 - confidence.gather(1, labels.unsqueeze(1)).squeeze()
            decisive_bonus = wrong_confidence.mean().item() * 0.5  # Was 2.0
            
            return base_reward + decisive_bonus
        else:
            # Attack failed - apply failure penalty
            return self.failure_penalty
    
    def _compute_imperceptibility_reward(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        attack_info: Dict[str, Any]
    ) -> float:
        """
        Compute reward for perturbation imperceptibility.
        
        Philosophy: 
        - Reward SSIM/PSNR directly without caring about epsilon budget
        - The agent should find the MINIMUM epsilon needed for success
        - Higher SSIM = higher reward, period.
        - PENALTY for unacceptably low quality (SSIM < 0.9 or PSNR < 35)
        """
        # Denormalize if necessary for correct metric computation
        if hasattr(self, 'normalize_stats') and self.normalize_stats is not None:
            mean = torch.tensor(self.normalize_stats.get('mean', [0.5])).view(1, -1, 1, 1).to(original.device)
            std = torch.tensor(self.normalize_stats.get('std', [0.5])).view(1, -1, 1, 1).to(original.device)
            orig_denorm = original * std + mean
            adv_denorm = adversarial * std + mean
        else:
            orig_denorm = original
            adv_denorm = adversarial
        
        # ============ SSIM CALCULATION ============
        ssim_val = 0.5  # default
        try:
            from skimage.metrics import structural_similarity
            orig_np = orig_denorm.squeeze().cpu().detach().numpy()
            adv_np = adv_denorm.squeeze().cpu().detach().numpy()
            
            orig_np = orig_np.clip(0, 1)
            adv_np = adv_np.clip(0, 1)
            
            if orig_np.ndim == 3:
                orig_np = orig_np.transpose(1, 2, 0)
                adv_np = adv_np.transpose(1, 2, 0)
            
            ssim_val = structural_similarity(
                orig_np, adv_np, 
                data_range=1.0,
                channel_axis=-1 if orig_np.ndim == 3 else None
            )
        except Exception as e:
            pass
        
        # ============ PSNR CALCULATION ============
        psnr_val = 30.0  # default
        try:
            mse = ((orig_denorm - adv_denorm) ** 2).mean().item()
            if mse > 1e-10:
                psnr_val = 10 * np.log10(1.0 / mse)
            else:
                psnr_val = 100.0  # perfect
        except:
            pass
        
        # ============ REWARD CALCULATION ============
        # Base reward: SSIM^2 (strongly favors high SSIM)
        ssim_reward = ssim_val ** 2
        
        # PSNR bonus: normalize 30-50 dB → 0-0.5
        psnr_bonus = max(0, min(0.5, (psnr_val - 30) / 40))
        
        # ============ PENALTIES FOR LOW QUALITY ============
        penalty = 0.0
        
        # SSIM penalty: for medical imaging, SSIM < 0.90 is unacceptable
        # With w_imperceptibility=0.8, we need penalty ~2.5 to make total reward ≈ failure
        if ssim_val < 0.85:
            penalty += 3.0  # Catastrophic - worse than failure
        elif ssim_val < 0.90:
            penalty += 2.5  # Equivalent to failure (total reward ≈ -0.1)
        elif ssim_val < 0.95:
            penalty += 0.3  # Light penalty, encourage higher
        
        # PSNR penalty: unacceptable if < 35 dB
        if psnr_val < 30:
            penalty += 1.0  # Strong penalty
        elif psnr_val < 35:
            penalty += 0.3  # Moderate penalty
        
        total_reward = ssim_reward + psnr_bonus - penalty
        
        return total_reward
    
    def _compute_efficiency_reward(
        self,
        attack_info: Dict[str, Any]
    ) -> float:
        """Compute reward for query efficiency.
        
        Uses log-scale for queries to not completely kill Square Attack
        (which needs ~1000 queries) while still rewarding PIXEL/FREQUENCY
        for using only ~40 queries.
        
        log(40)/log(5000) ≈ 0.43 → reward ≈ 0.57
        log(1000)/log(5000) ≈ 0.81 → reward ≈ 0.19
        log(5000)/log(5000) = 1.0 → reward = 0.0
        """
        queries_used = max(attack_info.get("queries", 1), 1)
        
        # Log-scale query efficiency (max 5000 queries)
        max_q = 5000
        query_reward = max(0.0, 1.0 - np.log(queries_used) / np.log(max_q))
        
        # Bonus for very efficient attacks (< 50 queries)
        if queries_used < 50:
            query_reward += 0.3
        
        return query_reward * 0.5  # Small contribution
    
    def _compute_diversity_reward(
        self,
        attack_info: Dict[str, Any]
    ) -> float:
        """
        Compute reward for attack diversity.
        
        Encourages exploration of different attack strategies and parameters.
        """
        strategy = attack_info.get("strategy", "unknown")
        
        # Update tracking
        self.strategy_counts[strategy] += 1
        self.recent_strategies.append(strategy)
        if len(self.recent_strategies) > self.diversity_window:
            self.recent_strategies.pop(0)
        
        # Track successes
        if attack_info.get("success", False):
            self.strategy_successes[strategy] += 1
        
        # Compute diversity metrics
        
        # 1. Strategy exploration bonus (reward rare strategies)
        total_attacks = sum(self.strategy_counts.values())
        strategy_freq = self.strategy_counts[strategy] / max(1, total_attacks)
        exploration_bonus = max(0, 0.5 - strategy_freq)  # Bonus for underexplored
        
        # 2. Recent diversity (entropy of recent strategies)
        if len(self.recent_strategies) > 10:
            unique_recent = len(set(self.recent_strategies))
            max_unique = min(4, len(self.recent_strategies))  # 4 strategies
            entropy_bonus = unique_recent / max_unique
        else:
            entropy_bonus = 0.5
        
        # 3. Parameter diversity (reward unusual parameter combinations)
        param_novelty = 0.0
        epsilon = attack_info.get("epsilon", 0.1)
        if epsilon < 0.05 or epsilon > 0.2:  # Unusual epsilon
            param_novelty += 0.2
        
        diversity_reward = exploration_bonus + 0.5 * entropy_bonus + param_novelty
        
        return diversity_reward * 0.1  # Very small contribution
    
    def _compute_confidence_reduction(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        attack_info: Dict[str, Any]
    ) -> float:
        """
        Compute shaped reward based on confidence reduction.
        
        DISABLED for adaptive epsilon training - we want clear success/failure signal.
        The agent should learn to find the right epsilon, not get partial credit.
        """
        # Return 0 - no partial credit for failed attacks
        return 0.0
    
    def _compute_curriculum_bonus(
        self,
        components: RewardComponents,
        curriculum_phase: int
    ) -> float:
        """
        Compute curriculum-based bonus.
        
        Different phases emphasize different objectives:
        - Phase 0: Focus on success (easier epsilon)
        - Phase 1: Balance success and efficiency
        - Phase 2: Full multi-objective
        - Phase 3: Emphasize imperceptibility
        """
        bonus = 0.0
        
        if curriculum_phase == 0:
            # Early training: small bonus for any success
            if components.success > 0:
                bonus = 0.2  # Was 2.0
                
        elif curriculum_phase == 1:
            # Efficiency matters more
            if components.success > 0 and components.efficiency > 0.5:
                bonus = 0.15  # Was 1.5
                
        elif curriculum_phase == 2:
            # Diversity matters
            if components.diversity > 0.5:
                bonus = 0.1  # Was 1.0
                
        elif curriculum_phase >= 3:
            # Full optimization: bonus for balanced performance
            if (components.success > 0 and 
                components.imperceptibility > 0.3 and
                components.efficiency > 0.3):
                bonus = 0.2  # Was 2.0
        
        return bonus
    
    def reset_episode(self):
        """Reset per-episode tracking."""
        pass
    
    def reset_diversity_tracking(self):
        """Reset diversity tracking (e.g., at start of new training phase)."""
        self.recent_strategies = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reward function statistics."""
        total_attacks = sum(self.strategy_counts.values())
        
        stats = {
            "total_attacks": total_attacks,
            "strategy_distribution": dict(self.strategy_counts),
            "strategy_success_rates": {},
            "training_step": self.training_step,
            "curriculum_phase": self.curriculum_phase
        }
        
        for strategy, count in self.strategy_counts.items():
            successes = self.strategy_successes.get(strategy, 0)
            stats["strategy_success_rates"][strategy] = successes / max(1, count)
        
        return stats
    
    def update_weights(
        self,
        success_weight: Optional[float] = None,
        imperceptibility_weight: Optional[float] = None,
        efficiency_weight: Optional[float] = None,
        diversity_weight: Optional[float] = None
    ):
        """Update reward weights dynamically."""
        if success_weight is not None:
            self.w_success = success_weight
        if imperceptibility_weight is not None:
            self.w_imperceptibility = imperceptibility_weight
        if efficiency_weight is not None:
            self.w_efficiency = efficiency_weight
        if diversity_weight is not None:
            self.w_diversity = diversity_weight


class MedicalImperceptibilityReward:
    """
    Specialized imperceptibility reward for medical imaging.
    
    Medical images have specific requirements:
    - Perturbations should not affect diagnostic regions
    - Noise patterns should match expected imaging artifacts
    - Contrast and intensity changes must be clinically acceptable
    """
    
    def __init__(
        self,
        modality: str = "xray",
        device: str = "cuda"
    ):
        """
        Initialize medical-specific reward.
        
        Args:
            modality: Imaging modality (xray, ct, mri, dermoscopy)
            device: Computation device
        """
        self.modality = modality
        self.device = device
        
        # Modality-specific thresholds
        self.thresholds = {
            "xray": {"max_intensity_change": 0.1, "max_contrast_change": 0.15},
            "ct": {"max_intensity_change": 0.05, "max_contrast_change": 0.1},
            "mri": {"max_intensity_change": 0.08, "max_contrast_change": 0.12},
            "dermoscopy": {"max_intensity_change": 0.15, "max_contrast_change": 0.2}
        }
        
        self.current_thresholds = self.thresholds.get(
            modality, 
            self.thresholds["xray"]
        )
    
    def compute(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute medical-specific imperceptibility score.
        
        Args:
            original: Original medical image
            adversarial: Adversarial image
            attention_mask: Optional mask for diagnostic regions
            
        Returns:
            Imperceptibility score [0, 1]
        """
        perturbation = adversarial - original
        
        # Global intensity change
        intensity_change = perturbation.mean().abs().item()
        intensity_score = max(0, 1 - intensity_change / self.current_thresholds["max_intensity_change"])
        
        # Contrast change
        orig_std = original.std().item()
        adv_std = adversarial.std().item()
        contrast_change = abs(adv_std - orig_std) / max(0.01, orig_std)
        contrast_score = max(0, 1 - contrast_change / self.current_thresholds["max_contrast_change"])
        
        # Region-specific penalty (if mask provided)
        region_score = 1.0
        if attention_mask is not None:
            # Higher penalty for perturbations in diagnostic regions
            masked_pert = (perturbation.abs() * attention_mask).sum() / attention_mask.sum()
            unmasked_pert = (perturbation.abs() * (1 - attention_mask)).sum() / (1 - attention_mask).sum()
            
            # Reward if perturbation is lower in diagnostic regions
            if masked_pert < unmasked_pert:
                region_score = 1.2  # Bonus
            else:
                region_score = 0.8 * (unmasked_pert / (masked_pert + 1e-8))
        
        # Combined score
        score = 0.4 * intensity_score + 0.4 * contrast_score + 0.2 * region_score
        
        return min(1.0, score)


class RobustRewardFunction(RewardFunction):
    """
    Reward function for attacking adversarially robust models.
    
    Key differences from standard RewardFunction:
    - Lower SSIM threshold (0.85 instead of 0.90)
    - Epsilon efficiency bonus when misclassification achieved
    - More weight on success, less on imperceptibility
    - Confidence reduction reward enabled for partial credit
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        normalize_stats: Optional[Dict[str, List[float]]] = None
    ):
        super().__init__(config, device, normalize_stats)
        
        # Override weights for robust setting
        reward_config = config.get("reward", {})
        weights = reward_config.get("weights", {})
        
        # More focus on success, less on imperceptibility
        self.w_success = weights.get("success", 1.5)
        self.w_imperceptibility = weights.get("imperceptibility", 0.1)
        self.w_efficiency = weights.get("efficiency", 0.02)
        self.w_diversity = weights.get("diversity", 0.02)
        self.w_epsilon_efficiency = weights.get("epsilon_efficiency", 0.1)  # Low priority for robust
        
        # Relaxed thresholds for robust models
        self.ssim_threshold = reward_config.get("ssim_threshold", 0.85)
        self.psnr_threshold = reward_config.get("psnr_threshold", 30.0)
        
    def compute_reward(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
        labels: torch.Tensor,
        model_output: torch.Tensor,
        attack_info: Dict[str, Any],
        curriculum_phase: int = 0
    ) -> RewardComponents:
        """
        Compute reward with relaxed constraints for robust models.
        """
        self.training_step += 1
        self.curriculum_phase = curriculum_phase
        
        components = RewardComponents()
        
        # Compute quality metrics
        ssim_val = self._compute_ssim(original, adversarial)
        psnr_val = self._compute_psnr(original, adversarial)
        
        # Check if attack caused misclassification
        predictions = model_output.argmax(dim=-1)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        misclassified = (predictions != labels).float().mean().item() > 0.5
        
        # Store in attack_info
        attack_info["ssim"] = ssim_val
        attack_info["psnr"] = psnr_val
        attack_info["misclassified"] = misclassified
        
        # =================================================================
        # SUCCESS REWARD - Relaxed threshold for robust models (0.85)
        # =================================================================
        quality_success = misclassified and ssim_val >= self.ssim_threshold
        
        if quality_success:
            # True success
            components.success = self.success_reward
            
            # Small SSIM bonus (less important for robust attacks)
            ssim_bonus = max(0, (ssim_val - self.ssim_threshold) / (1.0 - self.ssim_threshold))
            components.imperceptibility = ssim_bonus * 0.5
            
        elif misclassified:
            # Misclassified but SSIM below threshold - still partial success!
            # For robust models, misclassification is hard, so we reward it
            ssim_penalty = max(0, self.ssim_threshold - ssim_val)
            components.success = self.success_reward * 0.5 - ssim_penalty
            components.imperceptibility = 0.0
            
        else:
            # No misclassification - failure but give partial credit
            components.success = self.failure_penalty
            components.imperceptibility = 0.0
        
        # =================================================================
        # EPSILON EFFICIENCY - Reward for using lower epsilon IF misclassified
        # =================================================================
        if misclassified:
            epsilon_used = attack_info.get("epsilon", self.epsilon_budget)
            epsilon_min = attack_info.get("epsilon_min", self.epsilon_budget * 0.2)
            epsilon_max = attack_info.get("epsilon_max", self.epsilon_budget)
            
            if epsilon_max > epsilon_min:
                # Bonus = 1 when epsilon = epsilon_min, 0 when epsilon = epsilon_max
                epsilon_ratio = (epsilon_max - epsilon_used) / (epsilon_max - epsilon_min)
                epsilon_ratio = max(0.0, min(1.0, epsilon_ratio))
                components.epsilon_efficiency = epsilon_ratio
            else:
                components.epsilon_efficiency = 0.0
        else:
            components.epsilon_efficiency = 0.0
            
        # Store in attack_info for env
        attack_info["success"] = quality_success
        attack_info["quality_ok"] = ssim_val >= self.ssim_threshold
        
        # Efficiency reward
        if misclassified:  # Give efficiency reward for any misclassification
            components.efficiency = self._compute_efficiency_reward(attack_info)
        else:
            components.efficiency = 0.0
        
        # Diversity reward
        components.diversity = self._compute_diversity_reward(attack_info)
        
        # =================================================================
        # CONFIDENCE REDUCTION - Enabled for robust models
        # This gives partial credit even when attack fails
        # =================================================================
        if not misclassified:
            components.confidence_reduction = self._compute_confidence_reduction_robust(
                model_output, labels, attack_info
            )
        else:
            components.confidence_reduction = 0.0
        
        # Curriculum bonus
        components.curriculum_bonus = self._compute_curriculum_bonus(
            components, curriculum_phase
        )
        
        # Weighted total
        components.total = (
            self.w_success * components.success +
            self.w_imperceptibility * components.imperceptibility +
            self.w_efficiency * components.efficiency +
            self.w_diversity * components.diversity +
            self.w_epsilon_efficiency * components.epsilon_efficiency +
            components.confidence_reduction +
            components.curriculum_bonus
        )
        
        return components
    
    def _compute_confidence_reduction_robust(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        attack_info: Dict[str, Any]
    ) -> float:
        """
        Compute shaped reward based on confidence reduction.
        
        For robust models, we want to reward ANY progress toward misclassification,
        even if the attack ultimately fails.
        """
        confidence = F.softmax(model_output, dim=-1)
        
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # Get confidence on true class
        true_conf = confidence.gather(1, labels.unsqueeze(1)).squeeze()
        
        # Reward for reducing true-class confidence
        # true_conf=1.0 → reward=0, true_conf=0.5 → reward=0.25, true_conf=0.0 → reward=0.5
        conf_reduction_reward = (1.0 - true_conf.mean().item()) * 0.5
        
        # Extra bonus if confidence is below 0.5 (model is uncertain)
        if true_conf.mean().item() < 0.5:
            conf_reduction_reward += 0.2
        
        # Extra bonus if a wrong class has higher confidence than true class
        max_wrong_conf = confidence.clone()
        max_wrong_conf.scatter_(1, labels.unsqueeze(1), 0.0)
        max_wrong_conf = max_wrong_conf.max(dim=1)[0]
        
        if max_wrong_conf.mean().item() > true_conf.mean().item():
            conf_reduction_reward += 0.3
        
        return conf_reduction_reward


def get_reward_function(
    config: Dict[str, Any],
    device: str = "cuda",
    normalize_stats: Optional[Dict[str, List[float]]] = None,
    mode: str = "standard"
) -> RewardFunction:
    """
    Factory function to create appropriate reward function.
    
    Args:
        config: Reward configuration
        device: Computation device
        normalize_stats: Normalization statistics
        mode: "standard" for normal models, "robust" for adversarially trained models
        
    Returns:
        Appropriate RewardFunction instance
    """
    if mode == "robust":
        return RobustRewardFunction(config, device, normalize_stats)
    else:
        return RewardFunction(config, device, normalize_stats)