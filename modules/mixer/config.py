"""
Mixer configuration and track specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional


TrackKind = Literal['speech', 'music', 'background']


@dataclass
class TrackSpec:
    """
    Description of a single input track.
    
    Attributes:
        path: Absolute or relative path to an audio file
        kind: One of 'speech', 'music', or 'background'
        channel: Background sub-channel id (e.g., 'ch1', 'ch2', 'ch3')
        gain_db: Per-track gain override in dB (applied on top of category
                 and background-channel gains). Defaults to 0.0 dB.
    """

    path: str
    kind: TrackKind
    channel: Optional[str] = None
    gain_db: float = 0.0

    def resolve_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass
class MixerConfig:
    """Configuration for the audio mixer."""

    # Output audio settings
    output_sample_rate: int = 48000
    output_channels: int = 2
    output_format: str = 'wav'

    # Volume settings (dB)
    master_gain_db: float = 0.0
    speech_gain_db: float = 0.0
    music_gain_db: float = 0.0
    background_channel_gains_db: Dict[str, float] = field(default_factory=lambda: {
        'ch1': 0.0,
        'ch2': 0.0,
        'ch3': 0.0,
    })

    # Post-processing
    normalize: bool = True
    target_peak_dbfs: float = -1.0

    # Export
    overwrite: bool = True


