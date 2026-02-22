"""Dataset wrappers for medical imaging datasets."""

from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import medmnist
from medmnist import INFO

from .preprocessing import get_transforms


# Dataset information
DATASET_INFO = {
    "pathmnist": {
        "name": "PathMNIST",
        "task": "multi-class",
        "n_channels": 3,
        "n_classes": 9,
        "size": 28,
        "description": "Colon pathology classification",
    },
    "chestmnist": {
        "name": "ChestMNIST",
        "task": "multi-label",
        "n_channels": 1,
        "n_classes": 14,
        "size": 28,
        "description": "Chest X-ray multi-label classification",
    },
    "pneumoniamnist": {
        "name": "PneumoniaMNIST",
        "task": "binary-class",
        "n_channels": 1,
        "n_classes": 2,
        "size": 28,
        "description": "Pneumonia detection from chest X-ray",
    },
    "dermamnist": {
        "name": "DermaMNIST",
        "task": "multi-class",
        "n_channels": 3,
        "n_classes": 7,
        "size": 28,
        "description": "Dermatoscopic image classification",
    },
    "octmnist": {
        "name": "OCTMNIST",
        "task": "multi-class",
        "n_channels": 1,
        "n_classes": 4,
        "size": 28,
        "description": "Retinal OCT classification",
    },
    "bloodmnist": {
        "name": "BloodMNIST",
        "task": "multi-class",
        "n_channels": 3,
        "n_classes": 8,
        "size": 28,
        "description": "Blood cell classification",
    },
    "retinamnist": {
        "name": "RetinaMNIST",
        "task": "ordinal-regression",
        "n_channels": 3,
        "n_classes": 5,
        "size": 28,
        "description": "Diabetic retinopathy grading from fundus photography",
    },
    "breastmnist": {
        "name": "BreastMNIST",
        "task": "binary-class",
        "n_channels": 1,
        "n_classes": 2,
        "size": 28,
        "description": "Breast ultrasound tumor classification",
    },
    "tissuemnist": {
        "name": "TissueMNIST",
        "task": "multi-class",
        "n_channels": 1,
        "n_classes": 8,
        "size": 28,
        "description": "Kidney cortex tissue classification",
    },
    "organamnist": {
        "name": "OrganAMNIST",
        "task": "multi-class",
        "n_channels": 1,
        "n_classes": 11,
        "size": 28,
        "description": "Abdominal CT organ classification (axial view)",
    },
    "organcmnist": {
        "name": "OrganCMNIST",
        "task": "multi-class",
        "n_channels": 1,
        "n_classes": 11,
        "size": 28,
        "description": "Abdominal CT organ classification (coronal view)",
    },
    "organsmnist": {
        "name": "OrganSMNIST",
        "task": "multi-class",
        "n_channels": 1,
        "n_classes": 11,
        "size": 28,
        "description": "Abdominal CT organ classification (sagittal view)",
    },
}


def download_medmnist(data_dir: str = "./data/medmnist", datasets: Optional[List[str]] = None) -> None:
    """Download MedMNIST datasets.
    
    Args:
        data_dir: Directory to store datasets
        datasets: List of dataset names to download (default: all)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    
    for name in datasets:
        if name not in DATASET_INFO:
            print(f"Warning: Unknown dataset {name}, skipping")
            continue
            
        print(f"Downloading {name}...")
        DataClass = getattr(medmnist, INFO[name]["python_class"])
        
        # Download train, val, test splits
        for split in ["train", "val", "test"]:
            DataClass(split=split, root=str(data_dir), download=True)
        
        print(f"  ✓ {name} downloaded")
    
    print("All datasets downloaded successfully!")


class MedMNISTDataset(Dataset):
    """Wrapper for MedMNIST datasets with preprocessing."""
    
    def __init__(
        self,
        name: str,
        split: str = "train",
        root: str = "./data/medmnist",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        image_size: int = 224,
        native: bool = False,
    ):
        """Initialize MedMNIST dataset.
        
        Args:
            name: Dataset name (e.g., 'pathmnist', 'chestmnist')
            split: Data split ('train', 'val', 'test')
            root: Root directory for data
            transform: Image transforms
            target_transform: Label transforms
            download: Whether to download if not present
            image_size: Target image size (ignored if native=True)
            native: Use native 28×28 resolution without pretrained preprocessing
        """
        if name not in DATASET_INFO:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_INFO.keys())}")
        
        self.name = name
        self.info = DATASET_INFO[name]
        self.split = split
        self.image_size = image_size if not native else 28
        self.native = native
        
        # Get the appropriate MedMNIST class
        DataClass = getattr(medmnist, INFO[name]["python_class"])
        
        # Load dataset
        self.dataset = DataClass(
            split=split,
            root=root,
            download=download,
            transform=None,  # We'll apply our own transforms
            target_transform=target_transform,
        )
        
        # Set up transforms
        if transform is None:
            self.transform = get_transforms(
                image_size=image_size,
                n_channels=self.info["n_channels"],
                augment=(split == "train"),
                native=native,
            )
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            # Convert PIL Image or numpy array to tensor
            # Avoid torch.from_numpy() due to potential numpy availability issues
            image = np.array(image)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            # Convert to tensor using torch.tensor instead of torch.from_numpy
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Handle labels
        label = torch.tensor(label).squeeze()
        if label.dim() == 0:
            label = label.long()
        
        return image, label
    
    @property
    def n_classes(self) -> int:
        return self.info["n_classes"]
    
    @property
    def n_channels(self) -> int:
        return self.info["n_channels"]
    
    @property
    def task(self) -> str:
        return self.info["task"]


def get_dataset(
    name: str,
    split: str = "train",
    root: str = "./data/medmnist",
    image_size: int = 224,
    download: bool = True,
    native: bool = False,
    **kwargs,
) -> MedMNISTDataset:
    """Get a MedMNIST dataset.
    
    Args:
        name: Dataset name
        split: Data split
        root: Data directory
        image_size: Target image size (ignored if native=True)
        download: Whether to download
        native: Use native 28×28 resolution
        **kwargs: Additional arguments
        
    Returns:
        MedMNISTDataset instance
    """
    return MedMNISTDataset(
        name=name,
        split=split,
        root=root,
        image_size=image_size,
        download=download,
        native=native,
        **kwargs,
    )


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop incomplete last batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_subset(
    dataset: Dataset,
    n_samples: int,
    seed: int = 42,
) -> Subset:
    """Get a random subset of a dataset.
    
    Args:
        dataset: Source dataset
        n_samples: Number of samples to select
        seed: Random seed
        
    Returns:
        Subset of the dataset
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    return Subset(dataset, indices)
