"""
Emotions Module - Text Emotion Classification

This module provides emotion classification for text using RuBERT-tiny2 model
from HuggingFace fine-tuned for Russian emotion detection.
Model: seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions

It classifies text into 28 emotion categories from the GoEmotions dataset.

Example usage:
    >>> from modules.emotions import EmotionClassifier
    >>> classifier = EmotionClassifier()
    >>> result = classifier.predict("Я так счастлив сегодня!")
    >>> print(result['emotion'])  # e.g., 'joy'
    >>> print(result['confidence'])  # e.g., 0.85
"""

from .config import InferenceConfig, DEFAULT_CONFIG
from .inference import EmotionClassifier

__all__ = [
    'EmotionClassifier',
    'InferenceConfig',
    'DEFAULT_CONFIG',
]

__version__ = '2.0.0'

