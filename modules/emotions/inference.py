"""
Emotions Classifier Inference

This module provides inference class for emotion classification
using the RuBERT-tiny2 model from HuggingFace fine-tuned for Russian emotion detection.
Model: seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions
"""

from typing import Dict, List, Optional
import logging

from .config import InferenceConfig, DEFAULT_CONFIG


# Emotion labels mapping (28 classes from GoEmotions dataset)
EMOTION_LABELS = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

# Reverse mapping: label -> id
LABEL_TO_ID = {label: idx for idx, label in EMOTION_LABELS.items()}


class EmotionClassifier:
    """
    Emotion classifier using RuBERT-tiny2 model from HuggingFace.
    
    Model: seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions
    Classifies text into 28 emotion categories from the GoEmotions dataset.
    """
    
    def __init__(self, config: InferenceConfig = None):
        """
        Initialize the emotion classifier.
        
        Args:
            config: Inference configuration. If None, uses default config.
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("audiobook.emotions")
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the model from HuggingFace using transformers pipeline."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers library is required. "
                "Install it with: pip install transformers"
            )
        
        self.logger.info(f"Загрузка модели эмоций из HuggingFace: {self.config.model_name}")
        self.logger.info("Это может занять некоторое время при первом запуске...")
        
        # Determine device for pipeline
        # Transformers pipeline expects: -1 for CPU, 0+ for CUDA GPU, or device string
        device_map = self.config.device
        if device_map == "cpu":
            device_map = -1
        elif device_map == "cuda":
            device_map = 0  # Use first GPU
        elif device_map == "mps":
            # MPS might not be fully supported by transformers, fallback to CPU
            device_map = -1
        
        # Create pipeline for text classification
        self.pipeline = pipeline(
            "text-classification",
            model=self.config.model_name,
            device=device_map,
            return_all_scores=True,  # Return all emotion scores, not just top-1
        )
        
        self.logger.info("Модель эмоций загружена и готова к использованию")
    
    def predict(self, text: str) -> Dict:
        """
        Predict emotion for the input text.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary with prediction results:
            {
                'emotion': str,      # Top-1 emotion
                'confidence': float, # Top-1 probability
                'top5': [            # Top-5 predictions
                    {'emotion': str, 'prob': float},
                    ...
                ]
            }
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Run inference using pipeline
        # With return_all_scores=True, pipeline commonly returns a batch:
        # [ [{'label': 'emotion', 'score': 0.95}, ...] ] for single input
        # Normalize to a flat list of dicts: [{'label': str, 'score': float}, ...]
        results = self.pipeline(text)
        
        scores_list: List[dict] = []
        if isinstance(results, list) and len(results) > 0:
            first = results[0]
            if isinstance(first, list):
                # Typical case: batch dimension present
                scores_list = first
            elif isinstance(first, dict) and 'score' in first:
                # Some pipelines may return a flat list already
                scores_list = results  # type: ignore[assignment]
            else:
                # Attempt to flatten any nested lists/dicts
                for r in results:
                    if isinstance(r, dict) and 'score' in r:
                        scores_list.append(r)
                    elif isinstance(r, list):
                        for d in r:
                            if isinstance(d, dict) and 'score' in d:
                                scores_list.append(d)
        
        if not scores_list:
            # Fallback if pipeline returns unexpected format
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'top5': [{'emotion': 'neutral', 'prob': 0.0}],
            }
        
        # Sort by score (descending), guarding against missing keys
        sorted_scores = sorted(scores_list, key=lambda x: float(x.get('score', 0.0)), reverse=True)
        
        # Get top-k predictions
        top_k_predictions: List[Dict[str, float]] = []
        for item in sorted_scores[: self.config.top_k]:
            emotion_name = item.get('label', 'neutral')
            score = float(item.get('score', 0.0))
            top_k_predictions.append({'emotion': emotion_name, 'prob': score})
        
        # Top-1 prediction
        top_emotion = top_k_predictions[0] if top_k_predictions else {'emotion': 'neutral', 'prob': 0.0}
        
        return {
            'emotion': top_emotion['emotion'],
            'confidence': top_emotion['prob'],
            'top5': top_k_predictions,
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def get_all_emotions(self) -> List[str]:
        """Get list of all emotion labels."""
        return list(EMOTION_LABELS.values())

