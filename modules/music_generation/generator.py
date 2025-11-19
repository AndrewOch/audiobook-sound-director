"""Music generation using MusicGen model."""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Union
import scipy.io.wavfile

from .config import GeneratorConfig
from .model import MusicGenModel
from .utils import format_emotions_prompt, calculate_max_tokens


class MusicGenerator:
    """Generate background music based on emotions using MusicGen.
    
    This class provides an easy-to-use interface for generating music
    from emotion predictions. It handles model loading, prompt formatting,
    and audio generation.
    
    Example:
        >>> from modules.music_generation import MusicGenerator
        >>> 
        >>> generator = MusicGenerator()
        >>> emotions = [("joy", 0.85), ("contentment", 0.70)]
        >>> audio = generator.generate_from_emotions(emotions, duration_seconds=30)
        >>> generator.save_audio(audio, "output.wav")
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize the Music Generator.
        
        Args:
            config: Generator configuration. If None, uses default config.
        """
        self.config = config or GeneratorConfig()
        self.model_wrapper = MusicGenModel(self.config)
        
    def load_model(self):
        """Load the MusicGen model and processor."""
        if not self.model_wrapper.is_loaded():
            self.model_wrapper.load()
            
    def generate_from_emotions(
        self,
        emotions: List[Tuple[str, float]],
        duration_seconds: int = 22,
        custom_prompt: Optional[str] = None
    ) -> np.ndarray:
        """Generate music from emotion predictions.
        
        Args:
            emotions: List of tuples (emotion_name, probability)
            duration_seconds: Duration of music to generate in seconds
            custom_prompt: Custom prompt template (optional)
            
        Returns:
            Generated audio as numpy array (1D)
            
        Example:
            >>> emotions = [("joy", 0.85), ("sadness", 0.15)]
            >>> audio = generator.generate_from_emotions(emotions, duration_seconds=30)
        """
        # Ensure model is loaded
        if not self.model_wrapper.is_loaded():
            self.load_model()
            
        # Format prompt
        if custom_prompt:
            prompt = format_emotions_prompt(
                emotions, 
                duration_seconds, 
                base_prompt=custom_prompt
            )
        else:
            prompt = format_emotions_prompt(emotions, duration_seconds)
            
        return self.generate_from_prompt(prompt, duration_seconds)
    
    def generate_from_prompt(
        self,
        prompt: str,
        duration_seconds: int = 22
    ) -> np.ndarray:
        """Generate music from a text prompt.
        
        Args:
            prompt: Text description for music generation
            duration_seconds: Duration of music to generate in seconds
            
        Returns:
            Generated audio as numpy array (1D)
            
        Example:
            >>> prompt = "Calm and peaceful piano music"
            >>> audio = generator.generate_from_prompt(prompt, duration_seconds=30)
        """
        # Ensure model is loaded
        if not self.model_wrapper.is_loaded():
            self.load_model()
            
        print(f"Generating music with prompt: {prompt}")
        
        # Prepare inputs
        inputs = self.model_wrapper.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.model_wrapper.device) for k, v in inputs.items()}
        
        # Calculate max tokens
        max_tokens = calculate_max_tokens(
            duration_seconds, 
            self.config.tokens_per_second
        )
        
        # Generate audio
        print(f"Generating {duration_seconds}s audio ({max_tokens} tokens)...")
        try:
            with torch.no_grad():
                audio_values = self.model_wrapper.model.generate(
                    **inputs,
                    do_sample=self.config.do_sample,
                    guidance_scale=self.config.guidance_scale,
                    max_new_tokens=max_tokens
                )
        except Exception as e:
            # Fallback: if running on MPS and generation fails, retry on CPU
            if str(self.model_wrapper.device) == "mps":
                print(f"MPS generation failed, retrying on CPU: {e}")
                # Reload model on CPU in float32
                self.config.device = "cpu"
                self.model_wrapper = MusicGenModel(self.config)
                self.model_wrapper.load()
                # Move inputs to CPU
                inputs = {k: v.to(self.model_wrapper.device) for k, v in inputs.items()}
                with torch.no_grad():
                    audio_values = self.model_wrapper.model.generate(
                        **inputs,
                        do_sample=self.config.do_sample,
                        guidance_scale=self.config.guidance_scale,
                        max_new_tokens=max_tokens
                    )
            else:
                raise
        
        # Convert to numpy array (mono audio)
        audio_array = audio_values[0, 0].cpu().numpy()
        
        print(f"Music generated. Shape: {audio_array.shape}")
        return audio_array
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sampling_rate: Optional[int] = None
    ):
        """Save generated audio to WAV file.
        
        Args:
            audio: Audio array to save
            output_path: Path to save the WAV file
            sampling_rate: Sampling rate (uses model's rate if None)
            
        Example:
            >>> audio = generator.generate_from_emotions([("joy", 0.9)])
            >>> generator.save_audio(audio, "output.wav")
        """
        if sampling_rate is None:
            sampling_rate = self.model_wrapper.sampling_rate
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Конвертируем в поддерживаемый тип данных
        # scipy.io.wavfile.write поддерживает int16, int32, float32
        audio_processed = np.array(audio, copy=True)
        
        # Replace NaN/Inf with 0
        if not np.isfinite(audio_processed).all():
            audio_processed = np.nan_to_num(audio_processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure mono/stereo shape is sane
        if audio_processed.ndim > 2:
            # Flatten extra dims by averaging across channels
            audio_processed = np.mean(audio_processed, axis=tuple(range(1, audio_processed.ndim)))
        if audio_processed.ndim == 2:
            # If more than 2 channels, average to stereo; if 2 keep; if 1D handled below
            if audio_processed.shape[0] in (1, 2):
                # Convert (channels, samples) -> (samples,) by taking first channel for consistency
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
            # Hard clip to [-1, 1]
            audio_processed = np.clip(audio_processed, -1.0, 1.0)
        
        # Убеждаемся, что это float32 (поддерживается scipy)
        if audio_processed.dtype != np.float32:
            audio_processed = audio_processed.astype(np.float32)
        
        # Prefer saving as PCM_16 for browser compatibility
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(output_path), audio_processed, samplerate=int(sampling_rate), subtype='PCM_16', format='WAV')
        except Exception:
            # Fallback: int16 via scipy
            int16 = np.int16(np.clip(audio_processed * 32767.0, -32768, 32767))
            scipy.io.wavfile.write(
                str(output_path),
                rate=int(sampling_rate),
                data=int16
            )
        
        print(f"Audio saved to: {output_path}")
    
    def get_sampling_rate(self) -> int:
        """Get the audio sampling rate.
        
        Returns:
            Sampling rate in Hz
        """
        if not self.model_wrapper.is_loaded():
            self.load_model()
        return self.model_wrapper.sampling_rate

