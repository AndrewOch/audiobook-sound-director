"""
Speech generation utilities.

Currently exposes ElevenLabs client helpers.
"""

from .elevenlabs_client import (
    ElevenLabsClient,
    ElevenLabsConfig,
    get_elevenlabs_client,
    safe_filename,
    generate_speech_with_emotions,
)

__all__ = [
    "ElevenLabsClient",
    "ElevenLabsConfig",
    "get_elevenlabs_client",
    "safe_filename",
    "generate_speech_with_emotions",
]

