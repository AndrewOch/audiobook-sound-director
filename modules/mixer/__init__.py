"""
Audio Mixer

Provides utilities to combine multiple audio tracks (speech, music, and
background channels) into a single mixed file with configurable volume
controls per category, per background channel, and per track.

Example:
    >>> from modules.mixer import AudioMixer, MixerConfig, TrackSpec
    >>> tracks = [
    ...     TrackSpec(path="voice.wav", kind="speech"),
    ...     TrackSpec(path="music.wav", kind="music"),
    ...     TrackSpec(path="bg_ch1.wav", kind="background", channel="ch1"),
    ... ]
    >>> mixer = AudioMixer(MixerConfig())
    >>> output = mixer.mix(tracks, output_path="mixed.wav")
    >>> print(output)
"""

from .config import MixerConfig, TrackSpec
from .mixer import AudioMixer

__all__ = [
    'MixerConfig',
    'TrackSpec',
    'AudioMixer',
]

__version__ = '0.1.0'


