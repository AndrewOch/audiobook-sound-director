"""
Configuration for Foli Classifier

This module contains configuration classes and constants for the
foley sound classification inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# Project root and model paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "foli"

# Model artifact paths
CHECKPOINT_PATH = MODELS_DIR / "checkpoint.pt"
LABEL_SPACES_PATH = MODELS_DIR / "label_spaces.json"
ONNX_MODEL_PATH = MODELS_DIR / "encoder.onnx"
CLASSES_MAP_PATH = MODELS_DIR / "classes_map.csv"


@dataclass
class InferenceConfig:
    """Configuration for Foli classifier inference."""
    
    # Model configuration
    model_name: str = "cointegrated/rubert-tiny2"
    max_length: int = 256
    
    # Device configuration
    device: Optional[str] = None  # If None, will auto-detect (cuda > mps > cpu)
    
    # Inference settings
    use_fp16: bool = False  # Use mixed precision (only for CUDA)
    batch_size: int = 1  # Batch size for inference
    
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

