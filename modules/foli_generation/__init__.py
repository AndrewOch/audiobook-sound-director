"""
Foley Sound Generation using AudioLDM 2

This package provides text-to-audio generation for sound effects/foley
based on the AudioLDM2 pipeline.

Example:
    >>> from modules.foli_generation import FoliGenerator, FoliGenConfig
    >>> gen = FoliGenerator(FoliGenConfig(audio_length_in_s=6.0))
    >>> audio = gen.generate("The sound of a doorbell in a quiet room")
    >>> gen.save_audio(audio, "doorbell.wav")
"""

from .config import FoliGenConfig
from .model import AudioLDM2Model
from .generator import FoliGenerator

__all__ = [
    'FoliGenConfig',
    'AudioLDM2Model',
    'FoliGenerator',
    'generate',
    'run',
]

__version__ = '0.1.0'


# --- Pipeline-compatible entry points ---
from pathlib import Path
from typing import Dict, Union

import json


def _label_to_prompt(label: str) -> str:
    clean = str(label).replace('_', ' ').lower()
    return f"The sound of {clean}. High quality, clear."


def _build_channel_prompt(preds: Dict) -> str:
    # preds like {'class': 'Doorbell', 'prob': 0.6, 'top5': [...]} from pipeline classifier
    top = preds.get('class') or (
        preds.get('top5', [{}])[0].get('class') if preds.get('top5') else None
    )
    if not top:
        return "A generic ambient background sound"
    return _label_to_prompt(top)


def generate(job_dir: Union[str, Path]) -> Dict[str, Path]:
    """Generate foley background channels based on classifier predictions.

    Reads foli_predictions.json from job_dir and generates three wav files:
    foli_ch1.wav, foli_ch2.wav, foli_ch3.wav
    """
    job_dir = Path(job_dir)
    preds_path = job_dir / "foli_predictions.json"

    # Load predictions if available
    preds = None
    if preds_path.exists():
        try:
            with open(preds_path, "r", encoding="utf-8") as f:
                preds = json.load(f)
        except Exception:
            preds = None

    gen = FoliGenerator(FoliGenConfig())

    results: Dict[str, Path] = {}
    for ch in ("ch1", "ch2", "ch3"):
        p = _build_channel_prompt(preds.get(ch, {})) if isinstance(preds, dict) else "Ambient background noise"
        audio = gen.generate(
            prompt=p,
            audio_length_in_s=gen.config.audio_length_in_s,
            negative_prompt=gen.config.negative_prompt or "Low quality.",
            num_inference_steps=gen.config.num_inference_steps,
            num_waveforms_per_prompt=1,
            seed=0,
        )
        out_path = job_dir / f"foli_{ch}.wav"
        gen.save_audio(audio, out_path)
        results[ch] = out_path

    return results


def run(job_dir: Union[str, Path]):
    return generate(job_dir)


