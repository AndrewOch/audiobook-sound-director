"""
Configuration for Speech Recognition

This module contains configuration classes and constants for
speech recognition using Whisper.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# Импортируем функцию для получения пути кэша
try:
    from modules.cache_config import get_whisper_download_root
except ImportError:
    # Fallback если модуль не найден
    def get_whisper_download_root():
        return None


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "speech_recognition"


@dataclass
class RecognizerConfig:
    """Configuration for Speech Recognizer."""
    
    # Whisper model size
    # Options: 'tiny', 'base', 'small', 'medium', 'large'
    model_size: Literal['tiny', 'base', 'small', 'medium', 'large'] = 'small'
    
    # Language (None for auto-detection)
    language: Optional[str] = 'ru'
    
    # Task type
    task: Literal['transcribe', 'translate'] = 'transcribe'
    
    # Word-level timestamps
    word_timestamps: bool = True
    
    # Transcription parameters
    temperature: float = 0.0  # Higher = more random
    beam_size: int = 1  # Beam search width (lower = faster)
    best_of: int = 1  # Number of candidates (lower = faster)
    patience: float = 1.0  # Beam search patience
    
    # Output options
    verbose: bool = False  # Print progress
    
    # Device configuration
    device: Optional[str] = None  # If None, will auto-detect (cuda > cpu)
    
    # Download options
    download_root: Optional[str] = None  # Custom model cache directory
    
    def __post_init__(self):
        """Auto-detect device if not specified (CUDA > MPS > CPU)."""
        if self.device is None:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        
        # Устанавливаем download_root на внешний диск, если не указан явно
        if self.download_root is None:
            whisper_cache = get_whisper_download_root()
            if whisper_cache:
                self.download_root = str(whisper_cache)


# Default configuration instance
DEFAULT_CONFIG = RecognizerConfig()


# Model information
MODEL_INFO = {
    'tiny': {
        'params': '39M',
        'ram': '~1GB',
        'speed': 'Very Fast',
        'quality': 'Low'
    },
    'base': {
        'params': '74M',
        'ram': '~1GB',
        'speed': 'Fast',
        'quality': 'Good'
    },
    'small': {
        'params': '244M',
        'ram': '~2GB',
        'speed': 'Medium',
        'quality': 'Better'
    },
    'medium': {
        'params': '769M',
        'ram': '~5GB',
        'speed': 'Slow',
        'quality': 'Very Good'
    },
    'large': {
        'params': '1550M',
        'ram': '~10GB',
        'speed': 'Very Slow',
        'quality': 'Best'
    }
}

