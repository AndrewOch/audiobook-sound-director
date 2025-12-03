"""
Foli Classifier Inference

This module provides inference classes for the foley sound classifier
using both PyTorch and ONNX Runtime backends.
"""

import json
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import (
    CHECKPOINT_PATH,
    LABEL_SPACES_PATH,
    ONNX_MODEL_PATH,
    InferenceConfig,
    DEFAULT_CONFIG,
)
from .model import MultiHeadClassifier, mean_pooling


class FoliClassifierBase(ABC):
    """Base class for Foli classifier inference."""
    
    def __init__(self, config: InferenceConfig = None):
        """
        Initialize the classifier.
        
        Args:
            config: Inference configuration. If None, uses default config.
        """
        self.config = config or DEFAULT_CONFIG
        self.label_spaces = self._load_label_spaces()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    
    def _load_label_spaces(self) -> dict:
        """Load label spaces from JSON file."""
        if not LABEL_SPACES_PATH.exists():
            raise FileNotFoundError(
                f"Label spaces file not found: {LABEL_SPACES_PATH}. "
                f"Please ensure models are in the correct location."
            )
        
        with open(LABEL_SPACES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @abstractmethod
    def predict(self, text: str) -> Dict:
        """
        Predict foley classes for the input text.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary with predictions for each channel:
            {
                'ch1': {
                    'class': str,  # Top-1 class
                    'prob': float,  # Top-1 probability
                    'top5': [
                        {'class': str, 'prob': float},
                        ...
                    ]
                },
                'ch2': {...},
                'ch3': {...}
            }
        """
        pass
    
    def _format_predictions(self, logits_dict: dict) -> Dict:
        """
        Format logits into prediction dictionary with top-k classes.
        
        Args:
            logits_dict: Dictionary with logits for each channel
        
        Returns:
            Formatted predictions dictionary
        """
        result = {}
        
        for ch in ['ch1', 'ch2', 'ch3']:
            # Get probabilities using softmax
            if isinstance(logits_dict[ch], torch.Tensor):
                probs = torch.softmax(logits_dict[ch], dim=-1).squeeze(0).detach().cpu().numpy()
            else:
                # Already numpy array (from ONNX)
                probs = logits_dict[ch].squeeze(0)
                # Apply softmax manually
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()
            
            # Get top-k indices
            top_k_indices = np.argsort(probs)[-self.config.top_k:][::-1]
            
            # Get top-1
            top_idx = int(top_k_indices[0])
            id2label = self.label_spaces[ch]['id2label']
            top_class = id2label[str(top_idx)] if isinstance(id2label, dict) else id2label[top_idx]
            
            # Format top-k predictions
            top_k_predictions = []
            for idx in top_k_indices:
                idx = int(idx)
                class_name = id2label[str(idx)] if isinstance(id2label, dict) else id2label[idx]
                top_k_predictions.append({
                    'class': class_name,
                    'prob': float(probs[idx])
                })
            
            result[ch] = {
                'class': top_class,
                'prob': float(probs[top_idx]),
                'top5': top_k_predictions,
            }
        
        return result


class FoliClassifierPyTorch(FoliClassifierBase):
    """Foli classifier using PyTorch backend."""
    
    def __init__(self, config: InferenceConfig = None):
        """
        Initialize PyTorch-based classifier.
        
        Args:
            config: Inference configuration
        """
        super().__init__(config)
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self) -> MultiHeadClassifier:
        """Load the PyTorch model from checkpoint."""
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT_PATH}. "
                f"Please ensure models are in the correct location."
            )
        
        # Load checkpoint (PyTorch 2.6 defaults to weights_only=True)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        # Create model
        model = MultiHeadClassifier(
            base_name=self.config.model_name,
            label_spaces=self.label_spaces,
            dropout_p=0.1,
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.config.device)
        
        return model
    
    def predict(self, text: str) -> Dict:
        """
        Predict foley classes for the input text using PyTorch.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary with predictions for each channel
        """
        # Tokenize input
        encoding = self.tokenizer(
            str(text).lower(),
            padding=False,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        
        # Move to device
        encoding = {k: v.to(self.config.device) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type='cuda', dtype=torch.float16)
                if self.config.use_fp16 and self.config.device == 'cuda'
                else nullcontext()
            )
            
            with autocast_ctx:
                outputs = self.model(**encoding)
        
        # Format and return predictions
        return self._format_predictions(outputs['logits'])


class FoliClassifierONNX(FoliClassifierBase):
    """Foli classifier using ONNX Runtime backend."""
    
    def __init__(self, config: InferenceConfig = None):
        """
        Initialize ONNX-based classifier.
        
        Args:
            config: Inference configuration
        """
        super().__init__(config)
        self.session = self._load_onnx_session()
        self.classifier_heads = self._load_classifier_heads()
    
    def _load_onnx_session(self):
        """Load ONNX Runtime session."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with: pip install onnxruntime"
            )
        
        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {ONNX_MODEL_PATH}. "
                f"Please ensure models are in the correct location."
            )
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if self.config.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
        return session
    
    def _load_classifier_heads(self) -> dict:
        """Load classifier head weights from checkpoint."""
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT_PATH}. "
                f"Please ensure models are in the correct location."
            )
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Extract classifier head weights
        heads = {}
        for ch in ['ch1', 'ch2', 'ch3']:
            weight_key = f'classifiers.{ch}.weight'
            bias_key = f'classifiers.{ch}.bias'
            
            heads[ch] = {
                'weight': state_dict[weight_key].numpy(),
                'bias': state_dict[bias_key].numpy(),
            }
        
        return heads
    
    def predict(self, text: str) -> Dict:
        """
        Predict foley classes for the input text using ONNX Runtime.
        
        Args:
            text: Input text to classify
        
        Returns:
            Dictionary with predictions for each channel
        """
        # Tokenize input
        encoding = self.tokenizer(
            str(text).lower(),
            padding=False,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        
        # Prepare ONNX inputs
        onnx_inputs = {
            'input_ids': encoding['input_ids'].numpy(),
            'attention_mask': encoding['attention_mask'].numpy(),
        }
        
        # Run ONNX inference (encoder only)
        onnx_outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = onnx_outputs[0]
        
        # Apply mean pooling
        last_hidden_tensor = torch.from_numpy(last_hidden_state)
        attention_mask_tensor = encoding['attention_mask']
        pooled = mean_pooling(last_hidden_tensor, attention_mask_tensor).numpy()
        
        # Apply classifier heads
        logits = {}
        for ch in ['ch1', 'ch2', 'ch3']:
            weight = self.classifier_heads[ch]['weight']
            bias = self.classifier_heads[ch]['bias']
            logits[ch] = np.dot(pooled, weight.T) + bias
        
        # Format and return predictions
        return self._format_predictions(logits)

