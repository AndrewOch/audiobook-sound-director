"""
Emotions Module - Text Emotion Classification

This module provides emotion classification for text using an LSTM-based
neural network. It classifies text into 28 emotion categories from the
GoEmotions dataset.

Example usage:
    >>> from modules.emotions import EmotionClassifier
    >>> classifier = EmotionClassifier()
    >>> result = classifier.predict("Я так счастлив сегодня!")
    >>> print(result['emotion'])  # e.g., 'joy'
    >>> print(result['confidence'])  # e.g., 0.85
"""

from .config import InferenceConfig, DEFAULT_CONFIG
from .inference import EmotionClassifier
from .model import AmbientDirector

__all__ = [
    'EmotionClassifier',
    'InferenceConfig',
    'DEFAULT_CONFIG',
    'AmbientDirector',
]

__version__ = '1.0.0'

