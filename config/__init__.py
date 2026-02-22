"""Configuration management for MedSecure."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        OmegaConf DictConfig object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge two configurations, with override taking precedence.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base, override)


def get_default_config() -> DictConfig:
    """Get default configuration."""
    config_dir = Path(__file__).parent
    return load_config(config_dir / "default.yaml")


def save_config(config: DictConfig, path: str | Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
