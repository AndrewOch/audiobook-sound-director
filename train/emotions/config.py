"""
Training Configuration for Emotions Classifier

This module contains configuration classes for training the
AmbientDirector emotion classification model.
"""

from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    """Configuration for training the Emotions classifier."""
    
    # Model architecture
    vocab_size: int = 50000  # Will be determined from tokenizer
    embed_dim: int = 256
    hidden_dim: int = 256
    num_classes: int = 28
    dropout_p: float = 0.3
    
    # Tokenization
    max_length: int = 64
    padding_idx: int = 1
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 40
    
    # Optimizer
    optimizer: str = "adamw"  # adamw or adam
    
    # Learning rate scheduling (optional)
    use_scheduler: bool = False
    warmup_ratio: float = 0.1
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # Logging
    log_every_n_steps: int = 50
    
    # Checkpoint saving
    save_best_only: bool = True
    eval_every_n_epochs: int = 1
    
    # Random seed
    seed: int = 42
    
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1  # From remaining data after test split
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)


# Default training configuration
DEFAULT_TRAIN_CONFIG = TrainConfig()

