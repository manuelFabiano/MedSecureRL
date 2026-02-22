"""Image preprocessing and augmentation utilities."""

from typing import Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# ImageNet normalization (commonly used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Grayscale normalization
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]


def get_transforms(
    image_size: int = 224,
    n_channels: int = 3,
    augment: bool = False,
    normalize: bool = True,
    native: bool = False,
) -> transforms.Compose:
    """Get image transforms for training or evaluation.
    
    Args:
        image_size: Target image size
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        augment: Whether to apply augmentation
        normalize: Whether to normalize
        native: If True, keep native resolution (28×28) and skip RGB conversion
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    # Resize to target size (skip if native mode)
    if not native:
        transform_list.append(transforms.Resize((image_size, image_size)))
        # Convert grayscale to RGB if needed (for pretrained models)
        if n_channels == 1:
            transform_list.append(GrayscaleToRGB())
    
    # Augmentation (only for training)
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    # Normalize
    if normalize:
        if native:
            # Simple normalization for native resolution
            if n_channels == 1:
                transform_list.append(
                    transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD)
                )
            else:
                transform_list.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )
        else:
            transform_list.append(
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            )
    
    return transforms.Compose(transform_list)


class GrayscaleToRGB:
    """Convert grayscale tensor to RGB by repeating channels."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def normalize_batch(
    images: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> torch.Tensor:
    """Normalize a batch of images.
    
    Args:
        images: Batch of images [B, C, H, W] in [0, 1]
        mean: Channel means (default: ImageNet)
        std: Channel stds (default: ImageNet)
        
    Returns:
        Normalized images
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    
    return (images - mean) / std


def denormalize_batch(
    images: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> torch.Tensor:
    """Denormalize a batch of images.
    
    Args:
        images: Batch of normalized images [B, C, H, W]
        mean: Channel means used for normalization
        std: Channel stds used for normalization
        
    Returns:
        Denormalized images in [0, 1]
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    
    return images * std + mean


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def to_tensor(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to tensor.
    
    Args:
        array: Numpy array
        device: Target device
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array).float()
    return tensor.to(device)


def resize_batch(
    images: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a batch of images.
    
    Args:
        images: Batch of images [B, C, H, W]
        size: Target size (H, W)
        mode: Interpolation mode
        
    Returns:
        Resized images
    """
    return F.interpolate(images, size=size, mode=mode, align_corners=False)


def clamp_batch(
    images: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Clamp image values to valid range.
    
    Args:
        images: Batch of images
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped images
    """
    return torch.clamp(images, min_val, max_val)


def compute_image_statistics(images: torch.Tensor) -> dict:
    """Compute statistics for a batch of images.
    
    Args:
        images: Batch of images [B, C, H, W]
        
    Returns:
        Dictionary with mean, std, min, max per channel
    """
    return {
        "mean": images.mean(dim=(0, 2, 3)).tolist(),
        "std": images.std(dim=(0, 2, 3)).tolist(),
        "min": images.min().item(),
        "max": images.max().item(),
    }


def random_crop(
    images: torch.Tensor,
    crop_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Random crop a batch of images.
    
    Args:
        images: Batch of images [B, C, H, W]
        crop_size: Crop size (H, W)
        
    Returns:
        Cropped images and crop position (top, left)
    """
    _, _, h, w = images.shape
    crop_h, crop_w = crop_size
    
    top = torch.randint(0, h - crop_h + 1, (1,)).item()
    left = torch.randint(0, w - crop_w + 1, (1,)).item()
    
    cropped = images[:, :, top:top+crop_h, left:left+crop_w]
    return cropped, (top, left)


def add_gaussian_noise(
    images: torch.Tensor,
    std: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian noise to images.
    
    Args:
        images: Batch of images
        std: Noise standard deviation
        
    Returns:
        Noisy images
    """
    noise = torch.randn_like(images) * std
    return images + noise
