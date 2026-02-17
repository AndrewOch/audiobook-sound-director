"""
Speech generation utilities.

Exposes ElevenLabs and Yandex SpeechKit client helpers.
"""

from .elevenlabs_client import (
    ElevenLabsClient,
    ElevenLabsConfig,
    get_elevenlabs_client,
    safe_filename,
    generate_speech_with_emotions,
)
from .yandex_speechkit_client import (
    YandexSpeechKitClient,
    YandexSpeechKitConfig,
    get_yandex_speechkit_client,
)

__all__ = [
    "ElevenLabsClient",
    "ElevenLabsConfig",
    "get_elevenlabs_client",
    "safe_filename",
    "generate_speech_with_emotions",
    "YandexSpeechKitClient",
    "YandexSpeechKitConfig",
    "get_yandex_speechkit_client",
]

