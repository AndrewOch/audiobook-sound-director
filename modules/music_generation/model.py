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
        
        # Choose device (cuda > mps > cpu)
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        # Prefer half precision on GPU/MPS for speed
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        # Load model with desired dtype
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=dtype,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        
        print(f"Using device: {self.device}, dtype: {dtype}")
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

