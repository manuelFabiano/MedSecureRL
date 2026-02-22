"""
Generate qualitative figure for paper: 2 rows × 3 columns.

Shows examples from different datasets where agent naturally chooses
pixel vs frequency strategy.

Layout:
    Original  |  Perturbation (×10)  |  Adversarial
    [PathMNIST - PIXEL strategy]
    [BloodMNIST - FREQUENCY strategy]

Usage:
    python generate_qualitative_figure.py \
        --victim-models-dir checkpoints/victim \
        --agent results/agent/final_agent \
        --output-dir results/paper_figures
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataset, get_dataloader
from models import get_victim_model
from rl.hierarchical_env import HierarchicalAttackEnv, AttackStrategy
from rl.hierarchical_agent_sac import create_hierarchical_agent_sac
from attacks import ATTACK_REGISTRY
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Class names for datasets
CLASS_NAMES = {
    'pathmnist': ['Adipose', 'Background', 'Debris', 'Lymphocytes',
                  'Mucus', 'Smooth Muscle', 'Normal Mucosa', 'Cancer Stroma', 'Tumor'],
    'dermamnist': ['Actinic kerat.', 'Basal cell ca.', 'Benign kerat.',
                   'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular les.'],
    'bloodmnist': ['Basophil', 'Eosinophil', 'Erythroblast', 'Immature gran.',
                   'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet'],
}


def get_class_name(label, dataset):
    """Get human-readable class name."""
    classes = CLASS_NAMES.get(dataset, [])
    if label < len(classes):
        return classes[label]
    return f"Class {label}"


def sharpen_image(img, amount=0.5):
    """Apply unsharp mask sharpening to improve blurry upscaled images."""
    # Unsharp mask: original + amount * (original - blurred)
    blurred = gaussian_filter(img, sigma=1.0)
    sharpened = img + amount * (img - blurred)
    return np.clip(sharpened, 0, 1)


def denormalize(img_tensor, normalize_stats, apply_sharpening=True):
    """Convert normalized tensor to [0,1] numpy image."""
    img = img_tensor.clone().detach().cpu()
    if normalize_stats:
        mean = torch.tensor(normalize_stats['mean']).view(-1, 1, 1)
        std = torch.tensor(normalize_stats['std']).view(-1, 1, 1)
        img = img * std + mean
    img = img.clamp(0, 1)
    if img.dim() == 4:
        img = img.squeeze(0)
    img = img.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)  # Convert grayscale to RGB for consistency
    
    # Apply sharpening to reduce blur from upscaling
    if apply_sharpening:
        img = sharpen_image(img, amount=0.7)
    
    return img


def compute_perturbation_vis(orig_tensor, adv_tensor, normalize_stats, amplification=10.0):
    """Compute amplified perturbation for visualization."""
    orig = orig_tensor.clone().detach().cpu()
    adv = adv_tensor.clone().detach().cpu()
    
    if normalize_stats:
        mean = torch.tensor(normalize_stats['mean']).view(-1, 1, 1)
        std = torch.tensor(normalize_stats['std']).view(-1, 1, 1)
        orig = orig * std + mean
        adv = adv * std + mean
    
    pert = (adv - orig)
    
    # Amplify and shift to [0, 1] (0.5 = no change)
    pert_vis = 0.5 + amplification * pert
    pert_vis = pert_vis.clamp(0, 1)
    
    if pert_vis.dim() == 4:
        pert_vis = pert_vis.squeeze(0)
    pert_vis = pert_vis.permute(1, 2, 0).numpy()
    if pert_vis.shape[2] == 1:
        pert_vis = np.repeat(pert_vis, 3, axis=2)
    
    # Apply sharpening
    pert_vis = sharpen_image(pert_vis, amount=0.7)
    
    return pert_vis


def get_confidence(model, image):
    """Get model confidence."""
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        max_prob, pred = probs.max(1)
    return max_prob.item(), pred.item()


def run_medsecure_attack(agent, env, image, label, victim_model, device, normalize_stats):
    """
    Run MedSecure agent on a single image (agent chooses strategy).
    
    Returns detailed info about the attack.
    """
    orig_conf, orig_pred = get_confidence(victim_model, image)
    
    obs, info = env.reset(options={'image': image, 'label': label})
    original = env.current_image.clone()
    
    # Agent chooses strategy
    base_state = obs[:21]
    strategy_idx = agent.meta_controller.select_strategy(base_state, deterministic=True)
    env.set_strategy(strategy_idx)
    strategy_name = AttackStrategy(strategy_idx).name
    
    # Run attack
    done = False
    attack_params = None
    while not done:
        params, _ = agent.controllers[strategy_idx].predict(base_state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(params)
        done = terminated or truncated
        base_state = obs[:21]
        attack_params = info.get('params', {})
    
    adversarial = env.current_adversarial if env.current_adversarial is not None else original
    
    adv_conf, adv_pred = get_confidence(victim_model, adversarial)
    success = (orig_pred == label.item()) and (adv_pred != label.item())
    
    # L-inf in pixel space
    if normalize_stats:
        mean = torch.tensor(normalize_stats['mean']).view(1, -1, 1, 1).to(device)
        std = torch.tensor(normalize_stats['std']).view(1, -1, 1, 1).to(device)
        linf = ((adversarial * std + mean) - (original * std + mean)).abs().max().item()
    else:
        linf = (adversarial - original).abs().max().item()
    
    return {
        'adversarial': adversarial,
        'strategy': strategy_name,
        'success': success,
        'orig_pred': orig_pred,
        'adv_pred': adv_pred,
        'orig_conf': orig_conf,
        'adv_conf': adv_conf,
        'linf': linf,
        'params': attack_params,
    }


def load_model_and_env(victim_path, dataset, device, epsilon):
    """Load victim model and create environment for a dataset."""
    checkpoint = torch.load(victim_path, map_location=device)
    native = checkpoint.get('native', False)
    n_classes = checkpoint['n_classes']
    normalize_stats = checkpoint.get('normalize_stats', None)
    
    if normalize_stats is None:
        normalize_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    
    victim_model = get_victim_model(
        checkpoint['model_name'], num_classes=n_classes,
        checkpoint=victim_path, device=str(device)
    )
    victim_model.eval()
    
    test_dataset = get_dataset(dataset, split='test', native=native, download=True)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    train_dataset = get_dataset(dataset, split='train', native=native, download=True)
    train_loader = get_dataloader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    env = HierarchicalAttackEnv(
        victim_model=victim_model,
        attack_registry=ATTACK_REGISTRY,
        dataloader=train_loader,
        config={"max_steps_per_image": 5, "rl": {"action_space": {"epsilon_range": [0.001, epsilon]}}},
        device=str(device),
        mode="separated",
        allowed_strategies=["pixel", "frequency"],
        normalize_stats=normalize_stats
    )
    env.set_curriculum_config(epsilon_range=[0.001, epsilon], curriculum_phase=3)
    
    return victim_model, env, test_loader, normalize_stats


def find_example_with_strategy(agent, env, test_loader, victim_model, device, 
                                normalize_stats, target_strategy, max_search=300):
    """Find a successful attack where agent naturally chooses target_strategy."""
    searched = 0
    for images, labels in test_loader:
        if searched >= max_search:
            break
        
        images = images.to(device)
        labels = labels.to(device).squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        
        # Check if correctly classified
        with torch.no_grad():
            pred = victim_model(images).argmax(1)
        if pred.item() != labels.item():
            continue
        
        searched += 1
        
        result = run_medsecure_attack(agent, env, images, labels, victim_model, device, normalize_stats)
        
        if result['success'] and result['strategy'] == target_strategy:
            return images.clone(), labels.clone(), result
    
    return None, None, None


def create_figure(examples, output_path, amplification=10.0):
    """
    Create beautiful 2×3 figure for paper.
    
    examples: list of (image, label, result, dataset, normalize_stats) tuples
    """
    # Style setup
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.linewidth'] = 0.5
    
    # Colors
    COLORS = {
        'PIXEL': '#2E86AB',      # Blue
        'FREQUENCY': '#A23B72',  # Magenta/Purple
        'success': '#E94F37',    # Red for misclassification
        'text': '#1C1C1C',       # Dark gray
        'subtle': '#666666',     # Medium gray
    }
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(8, 6.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.01, hspace=0.35,
                           left=0.02, right=0.98, top=0.88, bottom=0.08)
    
    # Column titles
    col_titles = ['Original', 'Perturbation (×10)', 'Adversarial']
    
    for row, (image, label, result, dataset, normalize_stats) in enumerate(examples):
        # Prepare images
        orig_np = denormalize(image, normalize_stats)
        adv_np = denormalize(result['adversarial'], normalize_stats)
        pert_np = compute_perturbation_vis(image, result['adversarial'], normalize_stats, amplification)
        
        strategy = result['strategy']
        strategy_color = COLORS[strategy]
        
        # Get class names
        true_class = get_class_name(label.item(), dataset)
        pred_class = get_class_name(result['adv_pred'], dataset)
        
        # Parameters
        params = result['params']
        eps_used = params.get('epsilon', result['linf'])
        
        images_data = [orig_np, pert_np, adv_np]
        
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(images_data[col], interpolation='nearest')
            ax.axis('off')
            
            # Column titles (only first row)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight='bold', 
                            color=COLORS['text'], pad=8)
            
            # Original column: dataset + true class
            if col == 0:
                # Dataset badge
                dataset_display = dataset.replace('mnist', 'MNIST').replace('path', 'Path').replace('derma', 'Derma').replace('blood', 'Blood')
                ax.text(0.03, 0.97, dataset_display, transform=ax.transAxes,
                       fontsize=8, fontweight='bold', color='white',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=strategy_color, 
                                edgecolor='none', alpha=0.9))
                
                # True class below image
                ax.text(0.5, -0.08, f'{true_class}', transform=ax.transAxes,
                       fontsize=10, ha='center', color=COLORS['text'])
            
            # Perturbation column: show strategy badge and epsilon
            elif col == 1:
                # Strategy badge inside image (top-left, below title)
                ax.text(0.03, 0.97, strategy, transform=ax.transAxes,
                       fontsize=9, fontweight='bold', va='top',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=strategy_color, 
                                edgecolor='none', alpha=0.9))
                
                # Epsilon below image
                param_text = f'ε = {eps_used:.4f}'
                ax.text(0.5, -0.08, param_text, transform=ax.transAxes,
                       fontsize=10, ha='center', color=COLORS['subtle'])
            
            # Adversarial column: predicted class
            elif col == 2:
                # Predicted class (in red since misclassified)
                ax.text(0.5, -0.08, f'→ {pred_class}', transform=ax.transAxes,
                       fontsize=10, ha='center', fontweight='bold', color=COLORS['success'])
    
    # Save
    fig.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate qualitative figure for paper')
    parser.add_argument('--victim-models-dir', type=str, required=True,
                        help='Directory with victim model checkpoints')
    parser.add_argument('--agent', type=str, required=True,
                        help='Path to MedSecure agent')
    parser.add_argument('--pixel-dataset', type=str, default='pathmnist',
                        help='Dataset for pixel strategy example')
    parser.add_argument('--freq-dataset', type=str, default='bloodmnist',
                        help='Dataset for frequency strategy example')
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help='Maximum epsilon budget')
    parser.add_argument('--amplification', type=float, default=10.0,
                        help='Perturbation amplification')
    parser.add_argument('--output-dir', type=str, default='results/paper_figures')
    parser.add_argument('--max-search', type=int, default=300,
                        help='Maximum images to search')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random based on time)')
    
    args = parser.parse_args()
    
    # Randomize seed if not specified
    if args.seed is None:
        args.seed = int(time.time()) % 100000
        print(f"Using random seed: {args.seed}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    victim_dir = Path(args.victim_models_dir)
    
    print("="*60)
    print("GENERATING QUALITATIVE FIGURE")
    print("="*60)
    print(f"Device: {device}")
    print(f"Pixel dataset: {args.pixel_dataset}")
    print(f"Frequency dataset: {args.freq_dataset}")
    
    # Load agent (we'll reuse it for both datasets)
    print("\nLoading agent...")
    
    # First, load with pixel dataset to create agent
    pixel_victim_path = victim_dir / f"resnet18_{args.pixel_dataset}_best.pth"
    pixel_model, pixel_env, pixel_loader, pixel_norm = load_model_and_env(
        pixel_victim_path, args.pixel_dataset, device, args.epsilon
    )
    
    agent_config = {
        "sac_lr": 3e-4, "sac_buffer_size": 100000, "sac_batch_size": 256,
        "sac_learning_starts": 1000,
        "meta_lr": 1e-4, "meta_buffer_size": 10000, "meta_batch_size": 32,
        "meta_epsilon_start": 0.0, "meta_epsilon_min": 0.0, "meta_epsilon_decay": 1.0,
    }
    
    agent = create_hierarchical_agent_sac(pixel_env, agent_config, str(device))
    agent.load(str(args.agent))
    agent.meta_controller.epsilon = 0.0
    print(f"  Loaded from: {args.agent}")
    
    examples = []
    
    # 1. Find PIXEL example
    print(f"\nSearching for PIXEL example in {args.pixel_dataset}...")
    pixel_img, pixel_label, pixel_result = find_example_with_strategy(
        agent, pixel_env, pixel_loader, pixel_model, device, pixel_norm,
        target_strategy='PIXEL', max_search=args.max_search
    )
    
    if pixel_img is None:
        print("ERROR: Could not find PIXEL example!")
        return
    
    print(f"  Found! {get_class_name(pixel_label.item(), args.pixel_dataset)} → "
          f"{get_class_name(pixel_result['adv_pred'], args.pixel_dataset)}")
    examples.append((pixel_img, pixel_label, pixel_result, args.pixel_dataset, pixel_norm))
    
    # 2. Find FREQUENCY example (different dataset)
    print(f"\nSearching for FREQUENCY example in {args.freq_dataset}...")
    
    freq_victim_path = victim_dir / f"resnet18_{args.freq_dataset}_best.pth"
    freq_model, freq_env, freq_loader, freq_norm = load_model_and_env(
        freq_victim_path, args.freq_dataset, device, args.epsilon
    )
    
    freq_img, freq_label, freq_result = find_example_with_strategy(
        agent, freq_env, freq_loader, freq_model, device, freq_norm,
        target_strategy='FREQUENCY', max_search=args.max_search
    )
    
    if freq_img is None:
        print("WARNING: Could not find FREQUENCY example, trying PIXEL instead...")
        freq_img, freq_label, freq_result = find_example_with_strategy(
            agent, freq_env, freq_loader, freq_model, device, freq_norm,
            target_strategy='PIXEL', max_search=args.max_search
        )
    
    if freq_img is None:
        print("ERROR: Could not find any successful example!")
        return
    
    print(f"  Found! Strategy={freq_result['strategy']}, "
          f"{get_class_name(freq_label.item(), args.freq_dataset)} → "
          f"{get_class_name(freq_result['adv_pred'], args.freq_dataset)}")
    examples.append((freq_img, freq_label, freq_result, args.freq_dataset, freq_norm))
    
    # Create figure
    print("\nGenerating figure...")
    output_path = output_dir / 'fig_qualitative_examples'
    create_figure(examples, output_path, args.amplification)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for img, label, result, dataset, _ in examples:
        print(f"\n{dataset.upper()} - {result['strategy']}:")
        print(f"  {get_class_name(label.item(), dataset)} → {get_class_name(result['adv_pred'], dataset)}")
        print(f"  ε = {result['params'].get('epsilon', 0):.4f}")
    
    print("\n" + "="*60)
    print(f"Seed used: {args.seed}")
    print(f"Figure saved: {output_path.with_suffix('.pdf')}")
    print("="*60)


if __name__ == '__main__':
    main()