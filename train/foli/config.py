"""
Training Configuration for Foli Classifier

This module contains configuration classes for training the
foley sound classification model.
"""

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration for training the Foli classifier."""
    
    # Model configuration
    model_name: str = "cointegrated/rubert-tiny2"
    max_length: int = 256
    
    # Training hyperparameters
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.06
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Regularization
    dropout_p: float = 0.1
    label_smoothing: float = 0.02
    
    # Mixed precision training
    use_fp16: bool = True
    compile_model: bool = False
    
    # Loss weighting
    cls_loss_weight: float = 1.0
    head_weights: dict = None  # e.g., {'ch1': 0.7, 'ch2': 1.15, 'ch3': 1.15}
    
    # Class balancing
    use_class_weights: bool = True
    class_weight_beta: float = 0.999  # For effective number of samples
    class_weight_min: float = 0.5
    class_weight_max: float = 3.0
    
    # Prior distribution adjustment
    use_prior_adjustment: bool = True
    tau: dict = None  # e.g., {'ch1': 0.0, 'ch2': 0.5, 'ch3': 0.5}
    
    # Early stopping
    patience: int = 2
    
    # Logging
    log_every_n_steps: int = 50
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Set default values for dict fields if not provided."""
        if self.head_weights is None:
            self.head_weights = {'ch1': 0.7, 'ch2': 1.15, 'ch3': 1.15}
        
        if self.tau is None:
            self.tau = {'ch1': 0.0, 'ch2': 0.5, 'ch3': 0.5}
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)


# Default training configuration
DEFAULT_TRAIN_CONFIG = TrainConfig()

