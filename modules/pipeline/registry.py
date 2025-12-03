"""Global model registry for lazy initialization and reuse.

This module holds singleton instances of heavy models used across the
application and exposes getter functions to retrieve them. It also
provides a `warm_up()` function to proactively load the models at
application startup to avoid cold-start latencies.
"""

from __future__ import annotations

import threading
from typing import Optional
import logging


# These imports are intentionally inside functions in case some optional
# dependencies are not installed. We keep top-level types only.

_lock = threading.RLock()
_emotions = None
_foli = None
_music = None
_speech = None


def get_emotion_classifier():
    """Get a singleton instance of the EmotionClassifier.

    Preference order:
    1. LLM-based classifier via Scaleway (if configured via env vars).
    2. Local HuggingFace/RuBERT-based EmotionClassifier as fallback.
    """
    global _emotions
    with _lock:
        if _emotions is not None:
            return _emotions

        logger = logging.getLogger("audiobook.pipeline")

        # Try LLM-based classifier first (Scaleway API)
        try:
            from modules.emotions.llm_emotion_classifier import (
                LLMEmotionClassifier,
                build_llm_emotion_config_from_env,
            )

            llm_cfg = build_llm_emotion_config_from_env()
            if llm_cfg is not None:
                _emotions = LLMEmotionClassifier(llm_cfg)
                logger.info(
                    "Emotion classifier: using LLMEmotionClassifier (model=%s, project_id=%s)",
                    llm_cfg.model,
                    llm_cfg.project_id,
                )
                return _emotions
            else:
                logger.info(
                    "Emotion classifier: LLM config not found "
                    "(LLM_EMOTIONS_API_KEY / LLM_EMOTIONS_PROJECT_ID), "
                    "falling back to local EmotionClassifier"
                )
        except Exception as exc:
            logger.warning(
                "Emotion classifier: failed to initialize LLMEmotionClassifier, "
                "falling back to local EmotionClassifier: %s",
                exc,
            )

        # Fallback: local RuBERT-based classifier
        from modules.emotions.inference import EmotionClassifier

        _emotions = EmotionClassifier()
        logger.info("Emotion classifier: using local EmotionClassifier (HuggingFace model)")
        return _emotions


def get_foli_classifier():
    """Get a singleton instance of the Foli classifier (ONNX preferred)."""
    global _foli
    with _lock:
        if _foli is None:
            try:
                from modules.foli.inference import FoliClassifierONNX as _FoliCls
                _foli = _FoliCls()
            except Exception:
                from modules.foli.inference import FoliClassifierPyTorch as _FoliCls
                _foli = _FoliCls()
        return _foli


def get_music_generator():
    """Get a singleton instance of the MusicGenerator (preloaded)."""
    global _music
    with _lock:
        if _music is None:
            from modules.music_generation.generator import MusicGenerator
            _music = MusicGenerator()
            try:
                _music.load_model()
            except Exception:
                # Defer loading to first use if preload fails
                pass
        return _music


def get_speech_recognizer():
    """Get a singleton instance of the SpeechRecognizer."""
    global _speech
    with _lock:
        if _speech is None:
            from modules.speech_recognition.recognizer import SpeechRecognizer
            from modules.speech_recognition.config import RecognizerConfig
            # Default language None -> auto; language can be overridden per call
            # download_root будет автоматически установлен в __post_init__ из cache_config
            cfg = RecognizerConfig(language=None)
            _speech = SpeechRecognizer(cfg)
        return _speech


def warm_up() -> None:
    """Proactively load all heavy models to avoid cold-starts."""
    try:
        get_emotion_classifier()
    except Exception:
        pass
    try:
        get_foli_classifier()
    except Exception:
        pass
    try:
        get_music_generator()
    except Exception:
        pass
    try:
        get_speech_recognizer()
    except Exception:
        pass


