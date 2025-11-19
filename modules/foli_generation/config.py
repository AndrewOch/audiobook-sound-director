"""
Configuration for Foley Generation using AudioLDM2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FoliGenConfig:
    """
    Configuration for FoliGenerator.

    Attributes:
        repo_id: HuggingFace model repo id for AudioLDM2
        device: 'auto', 'cuda', 'mps', or 'cpu'
        use_fp16: If True and device is CUDA, use float16
        num_inference_steps: Number of diffusion steps (quality vs speed)
        audio_length_in_s: Length of generated audio in seconds
        negative_prompt: Optional negative prompt
        num_waveforms_per_prompt: Generate N candidates and pick best (0 index)
        sampling_rate: Output sampling rate (AudioLDM2 returns 16k)
    """

    repo_id: str = "cvssp/audioldm2"
    device: str = "auto"
    use_fp16: bool = True
    num_inference_steps: int = 50
    audio_length_in_s: float = 10.0
    negative_prompt: Optional[str] = None
    num_waveforms_per_prompt: int = 1
    sampling_rate: Optional[int] = 16000


