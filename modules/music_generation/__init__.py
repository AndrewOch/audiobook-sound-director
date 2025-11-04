"""Music Generation module for audiobook sound direction.

This module provides music generation capabilities using Meta's MusicGen model.
It generates background music based on emotion predictions from text.
"""

from .config import GeneratorConfig
from .generator import MusicGenerator
from .model import MusicGenModel
from .utils import format_emotions_prompt, calculate_max_tokens

__all__ = [
    "GeneratorConfig",
    "MusicGenerator",
    "MusicGenModel",
    "format_emotions_prompt",
    "calculate_max_tokens",
]

