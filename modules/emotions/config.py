"""
Configuration for Emotions Classifier

This module contains configuration classes and constants for the
emotion classification inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# Project root and model paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "emotions"

# Model artifact paths
CHECKPOINT_PATH = MODELS_DIR / "ambient_director.pt"
EMOTION_LABELS_PATH = MODELS_DIR / "emotion_labels.json"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"


@dataclass
class InferenceConfig:
    """Configuration for Emotions classifier inference."""
    
    # Model architecture parameters
    vocab_size: int = 50000  # Will be loaded from tokenizer
    embed_dim: int = 256
    hidden_dim: int = 256
    num_classes: int = 28
    dropout_p: float = 0.3
    
    # Tokenization parameters
    max_length: int = 64
    padding_idx: int = 1
    
    # Device configuration
    device: Optional[str] = None  # If None, will auto-detect (cuda > mps > cpu)
    
    # Output settings
    top_k: int = 5  # Number of top predictions to return
    
    def __post_init__(self):
        """Auto-detect device if not specified."""
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# Default configuration instance
DEFAULT_CONFIG = InferenceConfig()

