"""Victim model wrappers for adversarial attack testing."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


SUPPORTED_MODELS = ["resnet50", "resnet18", "densenet121", "efficientnet_b0", "vgg16", "simplecnn"]


class VictimModel(ABC, nn.Module):
    """Abstract base class for victim models."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._model = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels.
        
        Args:
            x: Input images
            
        Returns:
            Predicted labels [B]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input images
            
        Returns:
            Class probabilities [B, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def get_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction confidence.
        
        Args:
            x: Input images
            
        Returns:
            (max_probs, predicted_labels)
        """
        probs = self.predict_proba(x)
        max_probs, labels = probs.max(dim=1)
        return max_probs, labels
    
    def get_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            x: Input images
            y: True labels
            reduction: Loss reduction method
            
        Returns:
            Loss value
        """
        logits = self.forward(x)
        return F.cross_entropy(logits, y, reduction=reduction)
    
    def get_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient of loss w.r.t. input.
        
        Args:
            x: Input images (requires_grad=True)
            y: True labels
            
        Returns:
            Gradient tensor
        """
        x.requires_grad_(True)
        loss = self.get_loss(x, y)
        loss.backward()
        grad = x.grad.detach()
        x.requires_grad_(False)
        return grad
    
    def load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint.
        
        Handles various checkpoint formats:
        - Direct state_dict
        - Dict with "state_dict" key
        - Keys with/without "_model." prefix
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Check if keys need prefix adjustment
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # If there's already overlap, load directly
        if len(model_keys & ckpt_keys) > 0:
            self.load_state_dict(state_dict)
            return
        
        # No overlap - need to adjust prefixes
        sample_model_key = next(iter(model_keys))
        sample_ckpt_key = next(iter(ckpt_keys))
        
        # Determine what transformation is needed
        new_state_dict = {}
        
        if sample_model_key.startswith("_model."):
            # Model expects "_model." prefix
            if sample_ckpt_key.startswith("model."):
                # Checkpoint has "model." - replace with "_model."
                new_state_dict = {"_" + k: v for k, v in state_dict.items()}
            else:
                # Checkpoint has no prefix - add "_model."
                new_state_dict = {"_model." + k: v for k, v in state_dict.items()}
        else:
            # Model doesn't expect "_model." prefix
            if sample_ckpt_key.startswith("_model."):
                # Remove "_model." prefix
                new_state_dict = {k.replace("_model.", ""): v for k, v in state_dict.items()}
            elif sample_ckpt_key.startswith("model."):
                # Remove "model." prefix
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict
        
        self.load_state_dict(new_state_dict)
    
    def save_checkpoint(self, path: str) -> None:
        """Save model weights to checkpoint.
        
        Args:
            path: Output path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)


class ResNetVictim(VictimModel):
    """ResNet-based victim model.
    
    Supports both standard 224x224 inputs and small 28x28 inputs (for MedMNIST).
    """
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "resnet50",
        pretrained: bool = True,
        small_input: bool = False,
    ):
        """
        Args:
            num_classes: Number of output classes
            variant: ResNet variant (resnet18, resnet50, resnet101)
            pretrained: Use ImageNet pretrained weights
            small_input: If True, modify architecture for 28x28 inputs (no pretrained)
        """
        # If small_input, don't use pretrained weights (architecture changes)
        super().__init__(num_classes, pretrained and not small_input)
        self.variant = variant
        self.small_input = small_input
        
        # Load model
        if variant == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if (pretrained and not small_input) else None
            self._model = models.resnet50(weights=weights)
        elif variant == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and not small_input) else None
            self._model = models.resnet18(weights=weights)
        elif variant == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if (pretrained and not small_input) else None
            self._model = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
        
        # Modify for small inputs (28x28)
        if small_input:
            # Change first conv: kernel 7->3, stride 2->1, padding 3->1
            self._model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove max pooling (would reduce 28->14->7 too fast)
            self._model.maxpool = nn.Identity()
        
        # Replace classifier head
        in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class DenseNetVictim(VictimModel):
    """DenseNet-based victim model."""
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "densenet121",
        pretrained: bool = True,
    ):
        super().__init__(num_classes, pretrained)
        self.variant = variant
        
        # Load pretrained model
        if variant == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.densenet121(weights=weights)
        elif variant == "densenet169":
            weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.densenet169(weights=weights)
        else:
            raise ValueError(f"Unknown DenseNet variant: {variant}")
        
        # Replace classifier head
        in_features = self._model.classifier.in_features
        self._model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class EfficientNetVictim(VictimModel):
    """EfficientNet-based victim model."""
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "efficientnet_b0",
        pretrained: bool = True,
    ):
        super().__init__(num_classes, pretrained)
        self.variant = variant
        
        # Load pretrained model
        if variant == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.efficientnet_b0(weights=weights)
        elif variant == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.efficientnet_b1(weights=weights)
        elif variant == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.efficientnet_b2(weights=weights)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")
        
        # Replace classifier head
        in_features = self._model.classifier[1].in_features
        self._model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class VGGVictim(VictimModel):
    """VGG-based victim model."""
    
    def __init__(
        self,
        num_classes: int,
        variant: str = "vgg16",
        pretrained: bool = True,
    ):
        super().__init__(num_classes, pretrained)
        self.variant = variant
        
        # Load pretrained model
        if variant == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.vgg16(weights=weights)
        elif variant == "vgg19":
            weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self._model = models.vgg19(weights=weights)
        else:
            raise ValueError(f"Unknown VGG variant: {variant}")
        
        # Replace classifier head
        in_features = self._model.classifier[6].in_features
        self._model.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class SimpleCNN(VictimModel):
    """Lightweight CNN for native 28×28 medical images (no pretrained weights)."""
    
    def __init__(
        self,
        num_classes: int,
        n_channels: int = 3,
        **kwargs  # Ignore pretrained arg
    ):
        super().__init__(num_classes, pretrained=False)
        self.n_channels = n_channels
        
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_victim_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    small_input: bool = False,
) -> VictimModel:
    """Factory function to create victim models.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        checkpoint: Optional path to custom checkpoint
        device: Target device
        small_input: If True, modify architecture for 28x28 inputs (ResNet only)
        
    Returns:
        Initialized victim model
    """
    architecture = architecture.lower()
    
    # If checkpoint provided, try to read small_input from it
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location='cpu')
        small_input = ckpt.get('native', ckpt.get('small_input', small_input))
    
    if architecture in ["resnet50", "resnet18", "resnet101"]:
        model = ResNetVictim(num_classes, variant=architecture, pretrained=pretrained, small_input=small_input)
    elif architecture in ["densenet121", "densenet169"]:
        if small_input:
            raise ValueError(f"small_input not supported for {architecture}. Use simplecnn or resnet with --native.")
        model = DenseNetVictim(num_classes, variant=architecture, pretrained=pretrained)
    elif architecture in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]:
        if small_input:
            raise ValueError(f"small_input not supported for {architecture}. Use simplecnn or resnet with --native.")
        model = EfficientNetVictim(num_classes, variant=architecture, pretrained=pretrained)
    elif architecture in ["vgg16", "vgg19"]:
        if small_input:
            raise ValueError(f"small_input not supported for {architecture}. Use simplecnn or resnet with --native.")
        model = VGGVictim(num_classes, variant=architecture, pretrained=pretrained)
    elif architecture == "simplecnn":
        model = SimpleCNN(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: {SUPPORTED_MODELS}")
    
    # Load checkpoint if provided
    if checkpoint is not None:
        model.load_checkpoint(checkpoint)
    
    return model.to(device)


def freeze_batch_norm(model: nn.Module) -> None:
    """Freeze batch normalization layers.
    
    Args:
        model: Model to modify
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False