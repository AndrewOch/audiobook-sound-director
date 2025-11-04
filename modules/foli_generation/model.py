"""
AudioLDM2 Model Wrapper for Foley Generation.
"""

from __future__ import annotations

from typing import Optional

import torch

from .config import FoliGenConfig


class AudioLDM2Model:
    """Wrapper around Diffusers AudioLDM2Pipeline.

    Handles model loading, device/dtype selection, and pipeline access.
    """

    def __init__(self, config: Optional[FoliGenConfig] = None):
        self.config = config or FoliGenConfig()
        self.pipe = None
        self.device = None
        self.sampling_rate = self.config.sampling_rate or 16000

    def load(self):
        from diffusers import AudioLDM2Pipeline

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        # Choose dtype
        dtype = torch.float16 if (self.device == "cuda" and self.config.use_fp16) else torch.float32

        # Load pipeline
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            self.config.repo_id,
            dtype=dtype,
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # AudioLDM2 outputs audios at 16 kHz typically
        try:
            # Some versions expose feature extractor with sampling rate
            self.sampling_rate = int(getattr(self.pipe, 'feature_extractor').sampling_rate)
        except Exception:
            self.sampling_rate = self.sampling_rate or 16000

    def is_loaded(self) -> bool:
        return self.pipe is not None


