# MedSecure: Learning Adaptive Adversarial Attacks for Medical AI via Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code for the paper:

> **Learning Adaptive Adversarial Attacks for Medical AI via Reinforcement Learning**
> Manuel Fabiano, Fabio Orazio Mirto, Giovanni Merlino, Francesco Longo
> University of Messina, Italy
> *Under review at IEEE SMARTCOMP 2026*

---

## Abstract

Adversarial attacks on medical image classifiers typically use fixed hyperparameters, applying the same perturbation budget uniformly regardless of image difficulty. This is suboptimal: an intelligent attacker seeks the minimum perturbation sufficient for misclassification while minimizing queries to avoid detection. We present **MedSecure**, a hierarchical reinforcement learning framework that learns adaptive, per-image attack policies. A Double DQN meta-controller selects the attack strategy (pixel-domain or frequency-domain) while independent Soft Actor-Critic controllers optimize continuous parameters conditioned on each image. On MedMNIST benchmarks, MedSecure achieves **99.4% attack success rate** with significantly higher imperceptibility (**SSIM 0.997**) than fixed-budget baselines, by learning to exploit the minimum perturbation needed for each input. Notably, the policy is trained once on a single dataset and **transfers zero-shot** to unseen medical imaging domains, maintaining high success rates (≥ 97%) and image quality without retraining. The learned policy requires only **∼12 model queries per image**, enabling practical deployment in scenarios where excessive querying would trigger anomaly detection.

---

## Pre-trained Models & Results

Trained victim model checkpoints, trained RL agents, and experiment results are available on OneDrive:

📁 **[MedSecure — Checkpoints & Results](https://unimeit-my.sharepoint.com/:f:/r/personal/fmirto_unime_it/Documents/MedSecure?csf=1&web=1&e=mbaCcb)**

The folder contains:
- `checkpoints/victim/` — Trained victim model weights (ResNet18 on PathMNIST)
- `evaluation/hierarchical_agent_pathmnist/` — Trained RL agent checkpoint and evaluation results

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medsecure.git
cd medsecure

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

---

## Quick Start

### 1. Download Dataset
```bash
python -c "from data.datasets import download_medmnist; download_medmnist(datasets=['pathmnist'])"
```

### 2. Train Victim Model
```bash
python experiments/train_victim.py --model resnet18 --dataset pathmnist --epochs 20 --native
```

### 3. Run Baseline Attacks
```bash
python experiments/run_baselines.py \
    --model-path checkpoints/victim/resnet18_pathmnist_best.pth \
    --dataset pathmnist \
    --epsilon 0.03 \
    --output-dir results/baselines
```

### 4. Train RL Agent
```bash
python experiments/train_hierarchical_sac.py \
    --model-path checkpoints/victim/resnet18_pathmnist_best.pth \
    --dataset pathmnist \
    --timesteps 100000 \
    --output-dir results/agent
```

### 5. Evaluate
```bash
python experiments/evaluate.py \
    --agent results/agent/hierarchical_agent.pt \
    --victim-model checkpoints/victim/resnet18_pathmnist_best.pth \
    --dataset pathmnist \
    --n-samples 500 \
    --output-dir results/evaluation
```

Alternatively, run the full pipeline interactively with the **[MedSecure_Local.ipynb](MedSecure_Local.ipynb)** notebook.

---

## Project Structure

```
medsecure/
├── config/                       # Configuration files (default.yaml)
├── data/                         # MedMNIST dataset wrappers and preprocessing
├── models/                       # Victim model definitions (ResNet18, SimpleCNN)
├── attacks/                      # Attack implementations
│   ├── base_attack.py            # Abstract base class
│   ├── pixel_attack.py           # L∞ pixel perturbations
│   ├── frequency_attack.py       # DCT frequency-domain attack
│   ├── patch_attack.py           # Localized patch attack
│   ├── semantic_attack.py        # Color/brightness/contrast shift
│   ├── square_attack.py          # Square attack
│   └── baselines/                # FGSM, PGD, C&W, AutoAttack
├── rl/                           # RL components
│   ├── hierarchical_env.py       # Gym environment for attack selection
│   ├── hierarchical_agent_sac.py # Hierarchical DQN + SAC agent
│   ├── reward.py                 # Multi-objective reward function
│   ├── curriculum.py             # Curriculum learning scheduler
│   └── state_extractor.py        # Feature extraction for RL state
├── evaluation/                   # Metrics and visualization
│   ├── metrics.py                # ASR, SSIM, PSNR, LPIPS
│   ├── diversity.py              # Attack diversity metrics
│   ├── efficiency.py             # Query efficiency metrics
│   └── visualization.py          # Figures and plots
├── experiments/                  # Runnable scripts
│   ├── train_victim.py
│   ├── train_hierarchical_sac.py
│   ├── run_baselines.py
│   ├── evaluate.py
│   ├── run_paper_evaluation.py
│   ├── ablation.py
│   └── visualize_perturbations.py
├── scripts/                      # Shell scripts for reproducibility
├── tests/                        # Unit tests
├── MedSecure_Local.ipynb         # End-to-end demo notebook
└── requirements.txt
```

---


## Citation

If you use this code, please cite our paper:

```bibtex
@article{fabiano2026medsecure,
  title={Learning Adaptive Adversarial Attacks for Medical AI via Reinforcement Learning},
  author={Fabiano, Manuel and Mirto, Fabio Orazio and Merlino, Giovanni and Longo, Francesco},
  note={Under review at IEEE SMARTCOMP 2026},
  year={2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MedMNIST](https://medmnist.com/) dataset creators
- [PyTorch](https://pytorch.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) teams
- Adversarial robustness research community
