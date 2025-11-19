"""
Configuration for Emotions Classifier

This module contains configuration classes and constants for the
emotion classification inference using HuggingFace model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration for Emotions classifier inference."""
    
    # HuggingFace model name
    model_name: str = "seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions"
    
    # Number of emotion classes
    num_classes: int = 28
    
    # Device configuration
    device: Optional[str] = None  # If None, will auto-detect (cuda > mps > cpu)
    
    # Output settings
    top_k: int = 5  # Number of top predictions to return
    
    def __post_init__(self):
        """Auto-detect device if not specified."""
        if self.device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"


# Default configuration instance
DEFAULT_CONFIG = InferenceConfig()

