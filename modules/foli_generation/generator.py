"""
Foley sound generation using AudioLDM2.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import scipy.io.wavfile

from .config import FoliGenConfig
from .model import AudioLDM2Model


class FoliGenerator:
    def __init__(self, config: Optional[FoliGenConfig] = None, job_dir: Optional[Union[str, Path]] = None):
        self.config = config or FoliGenConfig()
        self.model = AudioLDM2Model(self.config)

    def load_model(self):
        if not self.model.is_loaded():
            self.model.load()

    def generate(
        self,
        prompt: str,
        audio_length_in_s: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        num_waveforms_per_prompt: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate foley audio from a text prompt.

        Returns mono audio at model sampling rate (default 16k) as numpy array.
        """
        if not self.model.is_loaded():
            self.load_model()

        params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt if negative_prompt is not None else self.config.negative_prompt,
            'audio_length_in_s': audio_length_in_s if audio_length_in_s is not None else self.config.audio_length_in_s,
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps,
            'num_waveforms_per_prompt': num_waveforms_per_prompt if num_waveforms_per_prompt is not None else self.config.num_waveforms_per_prompt,
        }

        # Seed control
        generator = None
        if seed is not None:
            device_for_gen = 'cuda' if self.model.device == 'cuda' else 'cpu'
            generator = torch.Generator(device_for_gen).manual_seed(int(seed))

        out = self.model.pipe(
            prompt=params['prompt'],
            negative_prompt=params['negative_prompt'],
            audio_length_in_s=params['audio_length_in_s'],
            num_inference_steps=params['num_inference_steps'],
            num_waveforms_per_prompt=params['num_waveforms_per_prompt'],
            generator=generator,
        )

        # AudioLDM2 returns list of audios; take the top ranked (index 0)
        audio = np.array(out.audios[0], dtype=np.float32)
        return audio

    def generate_batch(
        self,
        prompts: Sequence[str],
        audio_length_in_s: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        num_waveforms_per_prompt: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Generate multiple audios for multiple prompts.
        Returns a list of mono numpy arrays.
        """
        if not self.model.is_loaded():
            self.load_model()

        params = {
            'negative_prompt': negative_prompt if negative_prompt is not None else self.config.negative_prompt,
            'audio_length_in_s': audio_length_in_s if audio_length_in_s is not None else self.config.audio_length_in_s,
            'num_inference_steps': num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps,
            'num_waveforms_per_prompt': num_waveforms_per_prompt if num_waveforms_per_prompt is not None else self.config.num_waveforms_per_prompt,
        }

        generator = None
        if seed is not None:
            device_for_gen = 'cuda' if self.model.device == 'cuda' else 'cpu'
            generator = torch.Generator(device_for_gen).manual_seed(int(seed))

        out = self.model.pipe(
            prompt=list(prompts),
            negative_prompt=params['negative_prompt'],
            audio_length_in_s=params['audio_length_in_s'],
            num_inference_steps=params['num_inference_steps'],
            num_waveforms_per_prompt=params['num_waveforms_per_prompt'],
            generator=generator,
        )

        # Flatten to one best waveform per prompt
        # Diffusers returns len(prompts)*num_waveforms arrays ordered by prompt
        audios = [np.array(a, dtype=np.float32) for a in out.audios[: len(prompts)]]
        return audios

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sampling_rate: Optional[int] = None,
    ):
        """Save mono audio array to WAV file."""
        sr = sampling_rate or self.model.sampling_rate
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(output_path), rate=int(sr), data=audio)

    def get_sampling_rate(self) -> int:
        if not self.model.is_loaded():
            self.load_model()
        return int(self.model.sampling_rate)


