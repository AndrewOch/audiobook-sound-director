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
        
        # Конвертируем в поддерживаемый тип данных
        # scipy.io.wavfile.write поддерживает int16, int32, float32
        audio_processed = np.array(audio, copy=True)
        # Replace NaN/Inf with 0
        if not np.isfinite(audio_processed).all():
            audio_processed = np.nan_to_num(audio_processed, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure shape sane: convert to mono 1D
        if audio_processed.ndim > 2:
            audio_processed = np.mean(audio_processed, axis=tuple(range(1, audio_processed.ndim)))
        if audio_processed.ndim == 2:
            # Use first channel if stereo-like array is (channels, samples) or (samples, channels)
            if audio_processed.shape[0] in (1, 2):
                audio_processed = audio_processed[0]
            elif audio_processed.shape[1] in (1, 2):
                audio_processed = audio_processed[:,0]
            else:
                audio_processed = np.mean(audio_processed, axis=-1)
        
        # Конвертируем float16 в float32
        if audio_processed.dtype == np.float16:
            audio_processed = audio_processed.astype(np.float32)
        
        # Нормализуем значения в диапазон [-1.0, 1.0] для float32
        if audio_processed.dtype == np.float32:
            audio_processed = np.clip(audio_processed, -1.0, 1.0)
        
        # Убеждаемся, что это float32 (поддерживается scipy)
        if audio_processed.dtype != np.float32:
            audio_processed = audio_processed.astype(np.float32)
        
        # Prefer PCM_16
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(output_path), audio_processed, samplerate=int(sr), subtype='PCM_16', format='WAV')
        except Exception:
            int16 = np.int16(np.clip(audio_processed * 32767.0, -32768, 32767))
            scipy.io.wavfile.write(str(output_path), rate=int(sr), data=int16)

    def get_sampling_rate(self) -> int:
        if not self.model.is_loaded():
            self.load_model()
        return int(self.model.sampling_rate)


