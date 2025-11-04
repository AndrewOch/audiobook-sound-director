"""
Foli Module - Background Sound Classification

This module provides classification of background sounds (foley)
for audiobook production. It classifies text into three channels
of sound effects.

Example usage:
    >>> from modules.foli import FoliClassifierPyTorch
    >>> classifier = FoliClassifierPyTorch()
    >>> result = classifier.predict("Дверь хлопнула")
    >>> print(result['ch1']['class'])  # e.g., 'Door'
"""

from .config import InferenceConfig, DEFAULT_CONFIG
from .inference import FoliClassifierPyTorch, FoliClassifierONNX
from .model import MultiHeadClassifier, mean_pooling

__all__ = [
    'FoliClassifierPyTorch',
    'FoliClassifierONNX',
    'InferenceConfig',
    'DEFAULT_CONFIG',
    'MultiHeadClassifier',
    'mean_pooling',
]

__version__ = '1.0.0'

