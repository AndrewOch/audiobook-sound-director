"""
Emotions Classifier Inference

This module provides inference class for emotion classification
using the AmbientDirector LSTM-based model.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import (
    CHECKPOINT_PATH,
    EMOTION_LABELS_PATH,
    TOKENIZER_PATH,
    InferenceConfig,
    DEFAULT_CONFIG,
)
from .model import AmbientDirector


class EmotionClassifier:
    """
    Emotion classifier using AmbientDirector model.
    
    Classifies text into 28 emotion categories from the GoEmotions dataset.
    """
    
    def __init__(self, config: InferenceConfig = None):
        """
        Initialize the emotion classifier.
        
        Args:
            config: Inference configuration. If None, uses default config.
        """
        self.config = config or DEFAULT_CONFIG
        self.emotion_labels = self._load_emotion_labels()
        self.tokenizer = self._load_tokenizer()
        
        # Update vocab size from tokenizer
        self.config.vocab_size = self.tokenizer.get_vocab_size()
        
        self.model = self._load_model()
        self.model.eval()
    
    def _load_emotion_labels(self) -> dict:
        """Load emotion labels from JSON file."""
        if not EMOTION_LABELS_PATH.exists():
            raise FileNotFoundError(
                f"Emotion labels file not found: {EMOTION_LABELS_PATH}. "
                f"Please ensure models are in the correct location."
            )
        
        with open(EMOTION_LABELS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_tokenizer(self):
        """Load tokenizer from file."""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "tokenizers library is required. "
                "Install it with: pip install tokenizers"
            )
        
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {TOKENIZER_PATH}\n"
                f"Please download it using:\n"
                f"  python train/emotions/download_data.py\n"
                f"Or manually download from:\n"
                f"  https://drive.google.com/uc?id=1Atnpckju2lf31LU1li6HbGwIjp_pQRTK"
            )
        
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    
    def _load_model(self) -> AmbientDirector:
        """Load the model from checkpoint."""
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT_PATH}. "
                f"Please ensure the model file is in the correct location."
            )
        
        # Create model with config parameters
        model = AmbientDirector(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            dropout_p=self.config.dropout_p,
            padding_idx=self.config.padding_idx,
        )
        
        # Load state dict
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(self.config.device)
        
        return model
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            Tensor of token IDs with padding
        """
        # Encode text
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.config.max_length]
        
        # Pad to max_length
        ids += [self.config.padding_idx] * (self.config.max_length - len(ids))
        
        return torch.tensor([ids])
    
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
        # Tokenize input
        input_ids = self._tokenize(text).to(self.config.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[-self.config.top_k:][::-1]
        
        # Format results
        top_k_predictions = []
        for idx in top_k_indices:
            idx = int(idx)
            emotion_name = self.emotion_labels['id2label'][str(idx)]
            top_k_predictions.append({
                'emotion': emotion_name,
                'prob': float(probs[idx])
            })
        
        # Top-1 prediction
        top_emotion = top_k_predictions[0]
        
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
        return [
            self.emotion_labels['id2label'][str(i)]
            for i in range(self.config.num_classes)
        ]

