"""
Speech Recognition Module - Audio to Text with Timestamps

This module provides speech recognition capabilities using OpenAI's Whisper model.
It transcribes audio files and provides word-level timestamps.

Example usage:
    >>> from modules.speech_recognition import SpeechRecognizer
    >>> recognizer = SpeechRecognizer()
    >>> result = recognizer.transcribe("audio.mp3")
    >>> print(result['text'])
    >>> for word in result['segments'][0]['words']:
    ...     print(f"{word['text']}: {word['start']:.2f}s")
"""

from .config import RecognizerConfig, DEFAULT_CONFIG, MODEL_INFO
from .recognizer import SpeechRecognizer
from .model import WhisperModel
from .utils import (
    download_audio_from_url,
    download_audio_from_drive,
    validate_audio_file,
    get_audio_duration,
    format_timestamp,
    save_transcription_to_file,
)

__all__ = [
    'SpeechRecognizer',
    'RecognizerConfig',
    'DEFAULT_CONFIG',
    'MODEL_INFO',
    'WhisperModel',
    'download_audio_from_url',
    'download_audio_from_drive',
    'validate_audio_file',
    'get_audio_duration',
    'format_timestamp',
    'save_transcription_to_file',
]

__version__ = '1.0.0'

