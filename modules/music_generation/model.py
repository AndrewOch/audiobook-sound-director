"""Model wrapper for MusicGen."""

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from typing import Optional

from .config import GeneratorConfig


class MusicGenModel:
    """Wrapper around MusicgenForConditionalGeneration model.
    
    This class handles model loading, device management, and provides
    a clean interface for music generation.
    
    Attributes:
        config: Generator configuration
        model: MusicGen model instance
        processor: Audio processor for input preparation
        device: Device the model is running on
        sampling_rate: Audio sampling rate from model config
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize the MusicGen model.
        
        Args:
            config: Generator configuration. If None, uses default config.
        """
        self.config = config or GeneratorConfig()
        self.model = None
        self.processor = None
        self.device = None
        self.sampling_rate = None
        
    def load(self):
        """Load the model and processor from HuggingFace."""
        print(f"Loading MusicGen model: {self.config.model_name}")
        
        # Load model
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.config.model_name
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        
        # Set device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
            
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Get sampling rate from model config
        self.sampling_rate = self.model.config.audio_encoder.sampling_rate
        
        if self.config.sampling_rate is None:
            self.config.sampling_rate = self.sampling_rate
            
        print(f"Model loaded. Sampling rate: {self.sampling_rate} Hz")
        
    def is_loaded(self) -> bool:
        """Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.processor is not None

