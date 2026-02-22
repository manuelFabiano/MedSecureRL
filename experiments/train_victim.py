"""
Train victim models on MedMNIST datasets.

This script trains various CNN architectures (ResNet, DenseNet, EfficientNet)
on medical imaging datasets to serve as target models for adversarial attack testing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloader, get_dataset
from models import get_victim_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VictimTrainer:
    """
    Trainer for victim models on medical imaging datasets.
    """
    
    def __init__(
        self,
        model_name: str = 'resnet18',
        dataset_name: str = 'pathmnist',
        device: Optional[torch.device] = None,
        checkpoint_dir: str = 'checkpoints',
        native: bool = False,
        adversarial: bool = False,
        adv_epsilon: float = 0.01,
        adv_steps: int = 7,
        **kwargs
    ):
        """
        Args:
            model_name: Architecture to use
            dataset_name: MedMNIST dataset name
            device: Device for training
            checkpoint_dir: Directory for saving checkpoints
            native: Use native 28×28 resolution (for SimpleCNN)
            adversarial: Enable adversarial training (Madry et al.)
            adv_epsilon: PGD epsilon for adversarial training (pixel space)
            adv_steps: PGD steps for adversarial training
            **kwargs: Additional arguments for model/data
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.native = native
        self.adversarial = adversarial
        self.adv_epsilon = adv_epsilon
        self.adv_steps = adv_steps
        
        # Get dataset info
        train_dataset = get_dataset(dataset_name, split='train', native=native)
        self.n_classes = train_dataset.n_classes
        self.n_channels = train_dataset.n_channels
        
        # Create model (pass native as small_input for ResNet)
        self.model = get_victim_model(
            model_name,
            num_classes=self.n_classes,
            pretrained=kwargs.get('pretrained', True) and not native,  # No pretrained for native/small
            small_input=native  # Modify architecture for 28x28 if native
        )
        self.model = self.model.to(self.device)
        
        # Normalization stats (needed for adversarial training)
        if self.native:
            if self.n_channels == 1:
                self.normalize_stats = {'mean': [0.5], 'std': [0.5]}
            else:
                self.normalize_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        else:
            self.normalize_stats = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        
        self._norm_mean = torch.tensor(self.normalize_stats['mean']).view(1, -1, 1, 1).to(self.device)
        self._norm_std = torch.tensor(self.normalize_stats['std']).view(1, -1, 1, 1).to(self.device)
        
        if self.adversarial:
            logger.info(f"Adversarial training enabled: PGD-{adv_steps}, ε={adv_epsilon}")
        
        # Training state
        self.best_acc = 0.0
        self.history: Dict[str, list] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'cosine',
        early_stopping: int = 10,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the victim model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            scheduler_type: LR scheduler type ('cosine' or 'plateau')
            early_stopping: Epochs without improvement before stopping
            save_best: Whether to save best checkpoint
            verbose: Print progress
        
        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_dataset = get_dataset(self.dataset_name, split='train', native=self.native)
        val_dataset = get_dataset(self.dataset_name, split='val', native=self.native)
        
        train_loader = get_dataloader(
            train_dataset,
            batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = get_dataloader(
            val_dataset,
            batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Training loop
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update scheduler
            if scheduler_type == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_acc)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Check for improvement
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                epochs_without_improvement = 0
                
                if save_best:
                    self._save_checkpoint(epoch, val_acc, is_best=True)
            else:
                epochs_without_improvement += 1
            
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                    f"Best: {self.best_acc:.2%}"
                )
            
            # Early stopping
            if epochs_without_improvement >= early_stopping:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        return self.history
    
    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch (standard or adversarial)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            if self.adversarial:
                # Generate adversarial examples with PGD
                images_adv = self._pgd_attack(images, labels)
                # Train on adversarial examples (Madry et al.)
                optimizer.zero_grad()
                outputs = self.model(images_adv)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Track accuracy on adversarial (robust accuracy)
                _, predicted = outputs.max(1)
            else:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
            
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, correct / total
    
    def _pgd_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate adversarial examples with PGD for adversarial training.
        
        Follows Madry et al. (2018): PGD with random start, projecting
        in pixel space to enforce the L-inf budget correctly.
        
        Args:
            images: Batch of normalized images [B, C, H, W]
            labels: True labels [B]
            
        Returns:
            Adversarial images in normalized space
        """
        eps = self.adv_epsilon
        steps = self.adv_steps
        alpha = 2.5 * eps / steps  # Step size in pixel space
        
        # Convert alpha to normalized space
        alpha_norm = alpha / self._norm_std.mean().item()
        
        # Random start in normalized space
        eps_norm = eps / self._norm_std.mean().item()
        delta = torch.empty_like(images).uniform_(-eps_norm, eps_norm)
        delta = delta.detach()
        
        for _ in range(steps):
            delta.requires_grad_(True)
            
            outputs = self.model(images + delta)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            
            # Gradient ascent step
            grad = delta.grad.detach()
            delta = delta.detach() + alpha_norm * grad.sign()
            
            # Project in pixel space: denorm -> clamp -> renorm
            delta_pixel = delta * self._norm_std
            delta_pixel = torch.clamp(delta_pixel, -eps, eps)
            delta = delta_pixel / self._norm_std
            
            # Ensure valid image range
            x_adv = images + delta
            x_adv_denorm = x_adv * self._norm_std + self._norm_mean
            x_adv_denorm = torch.clamp(x_adv_denorm, 0.0, 1.0)
            x_adv = (x_adv_denorm - self._norm_mean) / self._norm_std
            delta = (x_adv - images).detach()
        
        return (images + delta).detach()
    
    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, correct / total
    
    def _save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """Save model checkpoint."""
        # Determine normalization stats based on native flag
        if self.native:
            if self.n_channels == 1:
                normalize_stats = {'mean': [0.5], 'std': [0.5]}
            else:
                normalize_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        else:
            # ImageNet normalization for pretrained models
            normalize_stats = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'n_classes': self.n_classes,
            'state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'history': self.history,
            'normalize_stats': normalize_stats,
            'native': self.native,
            'adversarial_training': self.adversarial,
            'adv_epsilon': self.adv_epsilon if self.adversarial else None,
            'adv_steps': self.adv_steps if self.adversarial else None,
        }
        
        filename = f"{self.model_name}_{self.dataset_name}"
        if self.adversarial:
            filename += "_robust"
        if is_best:
            filename += "_best"
        filename += ".pth"
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.best_acc = checkpoint.get('accuracy', 0)
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from {path} (accuracy: {self.best_acc:.2%})")
    
    @torch.no_grad()
    def evaluate(self, split: str = 'test') -> Dict[str, float]:
        """
        Evaluate on a dataset split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
        
        Returns:
            Dictionary with evaluation metrics
        """
        test_dataset = get_dataset(self.dataset_name, split=split, native=self.native)
        loader = get_dataloader(
            test_dataset,
            batch_size=64, shuffle=False, num_workers=4
        )
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        per_class_acc = {}
        for c in range(self.n_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                per_class_acc[c] = (all_preds[mask] == c).mean()
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'n_samples': total
        }


def train_victim_model(
    model_name: str = 'resnet18',
    dataset_name: str = 'pathmnist',
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    checkpoint_dir: str = 'checkpoints',
    native: bool = False,
    adversarial: bool = False,
    adv_epsilon: float = 0.01,
    adv_steps: int = 7,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train a victim model.
    """
    pretrained = kwargs.pop('pretrained', True)
    
    trainer = VictimTrainer(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_dir=checkpoint_dir,
        pretrained=pretrained,
        native=native,
        adversarial=adversarial,
        adv_epsilon=adv_epsilon,
        adv_steps=adv_steps,
    )
    
    history = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        **kwargs
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate('test')
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'best_val_acc': trainer.best_acc
    }


def main():
    """Main entry point for training victim models."""
    parser = argparse.ArgumentParser(description='Train victim models for MedSecure')
    
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'simplecnn'],
                       help='Model architecture (use simplecnn with --native)')
    parser.add_argument('--dataset', type=str, default='pathmnist',
                       choices=['pathmnist', 'chestmnist', 'dermamnist', 'octmnist',
                               'pneumoniamnist', 'bloodmnist'],
                       help='MedMNIST dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use pretrained weights')
    parser.add_argument('--native', action='store_true',
                       help='Use native 28×28 resolution (recommended with simplecnn)')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--adversarial', action='store_true',
                       help='Enable adversarial training (Madry et al.)')
    parser.add_argument('--adv-epsilon', type=float, default=0.01,
                       help='PGD epsilon for adversarial training (pixel space)')
    parser.add_argument('--adv-steps', type=int, default=7,
                       help='PGD steps for adversarial training')
    
    args = parser.parse_args()
    
    logger.info(f"Training {args.model} on {args.dataset}")
    logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    if args.native:
        logger.info("Using native 28×28 resolution (no resize, no pretrained preprocessing)")
    if args.adversarial:
        logger.info(f"Adversarial training: PGD-{args.adv_steps}, epsilon={args.adv_epsilon}")
    
    results = train_victim_model(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        pretrained=not args.no_pretrained,
        native=args.native,
        early_stopping=args.early_stopping,
        adversarial=args.adversarial,
        adv_epsilon=args.adv_epsilon,
        adv_steps=args.adv_steps,
    )
    
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.2%}")
    logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.2%}")


if __name__ == '__main__':
    main()