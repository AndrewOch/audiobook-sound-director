"""Configuration for Music Generation module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GeneratorConfig:
    """Configuration for Music Generator.
    
    Attributes:
        model_name: Name of the MusicGen model from HuggingFace
        device: Device to run the model on ('cuda', 'cpu', or 'auto')
        do_sample: Whether to use sampling during generation
        guidance_scale: Guidance scale for generation (higher = more adherence to prompt)
        tokens_per_second: Number of tokens generated per second of audio (default: 50)
        sampling_rate: Audio sampling rate in Hz
    """
    model_name: str = "facebook/musicgen-small"
    device: str = "auto"
    do_sample: bool = True
    guidance_scale: float = 3.0
    tokens_per_second: int = 50
    sampling_rate: Optional[int] = None  # Will be set from model config

