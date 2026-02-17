"""
Yandex SpeechKit TTS (API v1) client with per-sentence emotion support.

API docs: https://yandex.cloud/ru/docs/speechkit/tts/request
Endpoint: POST https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize

Authentication is performed via either an IAM token or an API key, both
passed through the ``Authorization`` header.  The client reads credentials
from environment variables at initialisation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io
import logging
import os
import re
import uuid

import requests

logger = logging.getLogger("audiobook.speech.yandex")


# ---------------------------------------------------------------------------
# Emotion mapping: GoEmotions (28 classes) → SpeechKit (emotion, speed)
# ---------------------------------------------------------------------------
# SpeechKit v1 supports emotion values: "good", "evil", "neutral"
# Only certain voices support all three (jane, zahar, ermil, alena, omazh).
# We also adjust speed to convey emotion through pacing.

EMOTION_TO_SPEECHKIT: Dict[str, Tuple[str, float]] = {
    # Positive / energetic → "good" emotion
    "joy":          ("good", 1.05),
    "love":         ("good", 0.95),
    "excitement":   ("good", 1.1),
    "amusement":    ("good", 1.05),
    "admiration":   ("good", 1.0),
    "approval":     ("good", 1.0),
    "caring":       ("good", 0.95),
    "gratitude":    ("good", 1.0),
    "optimism":     ("good", 1.0),
    "pride":        ("good", 1.0),
    "relief":       ("good", 0.95),
    "surprise":     ("good", 1.1),
    "desire":       ("good", 1.0),

    # Negative / aggressive → "evil" emotion
    "anger":        ("evil", 1.1),
    "annoyance":    ("evil", 1.05),
    "disgust":      ("evil", 0.95),
    "disapproval":  ("evil", 1.0),

    # Sad / heavy → "neutral" emotion + slower pace
    "sadness":      ("neutral", 0.85),
    "grief":        ("neutral", 0.8),
    "disappointment": ("neutral", 0.9),
    "remorse":      ("neutral", 0.85),
    "embarrassment": ("neutral", 0.9),

    # Tense / anxious → "neutral" + slightly faster
    "fear":         ("neutral", 1.05),
    "nervousness":  ("neutral", 1.05),
    "confusion":    ("neutral", 0.95),

    # Neutral / analytical
    "neutral":      ("neutral", 1.0),
    "curiosity":    ("neutral", 1.0),
    "realization":  ("neutral", 1.0),
}

# Voices that support emotion parameter (good / evil / neutral)
EMOTIONAL_VOICES = {"jane", "zahar", "ermil", "alena", "omazh"}

# Default voice — supports all three emotions
DEFAULT_VOICE = "jane"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class YandexSpeechKitConfig:
    """Settings for the SpeechKit TTS v1 endpoint.

    Exactly one of *iam_token* or *api_key* must be provided.
    *folder_id* is required when authenticating with an IAM token.
    """

    iam_token: Optional[str] = None
    api_key: Optional[str] = None
    folder_id: Optional[str] = None
    voice: str = DEFAULT_VOICE
    emotion: Optional[str] = None
    lang: str = "ru-RU"
    speed: float = 1.0
    format: str = "mp3"
    sample_rate_hertz: int = 48000
    base_url: str = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

    def __post_init__(self) -> None:
        if not self.iam_token and not self.api_key:
            raise ValueError(
                "Provide either iam_token or api_key for Yandex SpeechKit auth"
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class YandexSpeechKitClient:
    """Low-level wrapper around the SpeechKit TTS v1 REST API."""

    def __init__(self, config: YandexSpeechKitConfig) -> None:
        self.config = config
        self.session = requests.Session()

        if config.api_key:
            self.session.headers["Authorization"] = f"Api-Key {config.api_key}"
        else:
            self.session.headers["Authorization"] = f"Bearer {config.iam_token}"

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        lang: Optional[str] = None,
        speed: Optional[float] = None,
        format: Optional[str] = None,
        sample_rate_hertz: Optional[int] = None,
        ssml: bool = False,
    ) -> bytes:
        """Send a synthesis request and return raw audio bytes."""
        payload: dict[str, str] = {}

        if ssml:
            payload["ssml"] = text
        else:
            payload["text"] = text

        payload["lang"] = lang or self.config.lang
        payload["voice"] = voice or self.config.voice
        payload["speed"] = str(speed if speed is not None else self.config.speed)

        chosen_format = format or self.config.format
        payload["format"] = chosen_format

        if chosen_format == "lpcm":
            payload["sampleRateHertz"] = str(
                sample_rate_hertz or self.config.sample_rate_hertz
            )

        emo = emotion if emotion is not None else self.config.emotion
        if emo:
            payload["emotion"] = emo

        if self.config.folder_id:
            payload["folderId"] = self.config.folder_id

        resp = self.session.post(
            self.config.base_url,
            data=payload,
            timeout=120,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Yandex SpeechKit synthesis failed [%s]: %s",
                resp.status_code,
                resp.text,
            )
            raise exc

        return resp.content

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """Synthesize speech and write the result to *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_bytes = self.synthesize(text, **kwargs)
        with open(output_path, "wb") as fh:
            fh.write(audio_bytes)
        return output_path


# ---------------------------------------------------------------------------
# OAuth → IAM token exchange
# ---------------------------------------------------------------------------

IAM_TOKEN_URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"


def _exchange_oauth_for_iam_token(oauth_token: str) -> str:
    """Exchange Yandex OAuth token (y0_...) for IAM token."""
    resp = requests.post(
        IAM_TOKEN_URL,
        json={"yandexPassportOauthToken": oauth_token},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["iamToken"]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: Optional[YandexSpeechKitClient] = None


def get_yandex_speechkit_client() -> YandexSpeechKitClient:
    """Return (or create) a module-level singleton client.

    Environment variables
    ---------------------
    YANDEX_SPEECHKIT_API_KEY   – API-key (AQVN...) or OAuth token (y0_...).
    YANDEX_SPEECHKIT_IAM_TOKEN – IAM-token authentication (alternative).
    YANDEX_SPEECHKIT_FOLDER_ID – Folder ID (required for OAuth/IAM user auth).
    YANDEX_SPEECHKIT_VOICE     – Voice name (default ``jane``).
    YANDEX_SPEECHKIT_EMOTION   – Emotional tone / role.
    YANDEX_SPEECHKIT_LANG      – Language code (default ``ru-RU``).
    YANDEX_SPEECHKIT_SPEED     – Base speech rate 0.1–3.0 (default ``1.0``).
    YANDEX_SPEECHKIT_FORMAT    – Audio format (default ``mp3``).
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("YANDEX_SPEECHKIT_API_KEY")
    iam_token = os.getenv("YANDEX_SPEECHKIT_IAM_TOKEN")
    folder_id = os.getenv("YANDEX_SPEECHKIT_FOLDER_ID")

    if api_key and api_key.strip().startswith("y0_"):
        if not folder_id:
            raise RuntimeError(
                "OAuth token (y0_...) requires YANDEX_SPEECHKIT_FOLDER_ID."
            )
        iam_token = _exchange_oauth_for_iam_token(api_key.strip())
        api_key = None

    if not api_key and not iam_token:
        raise RuntimeError(
            "Set YANDEX_SPEECHKIT_API_KEY or YANDEX_SPEECHKIT_IAM_TOKEN"
        )

    config = YandexSpeechKitConfig(
        api_key=api_key,
        iam_token=iam_token,
        folder_id=folder_id,
        voice=os.getenv("YANDEX_SPEECHKIT_VOICE", DEFAULT_VOICE),
        emotion=os.getenv("YANDEX_SPEECHKIT_EMOTION"),
        lang=os.getenv("YANDEX_SPEECHKIT_LANG", "ru-RU"),
        speed=float(os.getenv("YANDEX_SPEECHKIT_SPEED", "1.0")),
        format=os.getenv("YANDEX_SPEECHKIT_FORMAT", "mp3"),
    )

    _client = YandexSpeechKitClient(config)
    return _client


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(
    r'(?<=[.!?…»])\s+|(?<=[.!?…»])\n',
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences keeping punctuation attached."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Local emotion classifier (lazy-loaded)
# ---------------------------------------------------------------------------

_emotion_clf = None


def _get_emotion_classifier():
    """Lazy-load the local RuBERT emotion classifier."""
    global _emotion_clf
    if _emotion_clf is not None:
        return _emotion_clf
    try:
        from modules.emotions.inference import EmotionClassifier
        _emotion_clf = EmotionClassifier()
        return _emotion_clf
    except Exception as exc:
        logger.warning("Cannot load emotion classifier: %s", exc)
        return None


def _classify_emotion(text: str) -> str:
    """Classify a single sentence and return the emotion label.

    If the top-1 is ``neutral`` with low confidence AND the runner-up is
    a real emotion with comparable score, prefer the runner-up — this makes
    short emotional sentences less likely to be stuck on ``neutral``.
    """
    clf = _get_emotion_classifier()
    if clf is None:
        return "neutral"

    pred = clf.predict(text)
    top5 = pred.get("top5", [])

    if len(top5) < 2:
        return pred.get("emotion", "neutral")

    top1_label = top5[0].get("emotion", "neutral")
    top1_score = float(top5[0].get("prob", 0.0))
    top2_label = top5[1].get("emotion", "neutral")
    top2_score = float(top5[1].get("prob", 0.0))

    if top1_label == "neutral" and top2_label != "neutral":
        if top1_score < 0.45 or (top1_score - top2_score) < 0.15:
            return top2_label

    return top1_label


def _map_emotion(label: str) -> Tuple[str, float]:
    """Map a GoEmotions label to (speechkit_emotion, speed_multiplier)."""
    return EMOTION_TO_SPEECHKIT.get(label, ("neutral", 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(prefix: str = "speech", suffix: str = ".mp3") -> str:
    """Generate a collision-free file name."""
    return f"{prefix}_{uuid.uuid4().hex}{suffix}"


# ---------------------------------------------------------------------------
# Main generation with per-sentence emotion
# ---------------------------------------------------------------------------

def generate_speech_with_emotions(
    text: str,
    job_dir: Path,
    *,
    voice: Optional[str] = None,
    emotion: Optional[str] = None,
    filename: str = "speech.wav",
) -> Path:
    """Generate emotional speech via Yandex SpeechKit.

    Pipeline
    --------
    1. Split *text* into individual sentences.
    2. Classify each sentence with the local RuBERT emotion model.
    3. Map detected emotion → SpeechKit ``emotion`` param + speed adjustment.
    4. Synthesize each sentence separately with its own emotion & speed.
    5. Concatenate all audio chunks into a single normalised WAV (48 kHz,
       stereo) and save it as *filename* inside *job_dir*.
    """
    from pydub import AudioSegment  # type: ignore

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    client = get_yandex_speechkit_client()
    chosen_voice = voice or client.config.voice
    base_speed = client.config.speed
    fmt = client.config.format

    supports_emotion = chosen_voice in EMOTIONAL_VOICES

    raw_text = (text or "").strip()
    if not raw_text:
        raise ValueError("Cannot generate speech for empty text")

    sentences = _split_sentences(raw_text)
    if not sentences:
        sentences = [raw_text]

    logger.info(
        "Emotional speech synthesis: %d sentences, voice=%s, emotions=%s",
        len(sentences),
        chosen_voice,
        "yes" if supports_emotion else "no (voice unsupported)",
    )

    combined = AudioSegment.empty()
    short_pause = AudioSegment.silent(duration=250)

    for idx, sentence in enumerate(sentences):
        emo_label = _classify_emotion(sentence)
        sk_emotion, speed_mult = _map_emotion(emo_label)
        effective_speed = round(base_speed * speed_mult, 2)
        effective_speed = max(0.1, min(3.0, effective_speed))

        logger.debug(
            "  [%d/%d] emotion=%s → sk=%s speed=%.2f | %s",
            idx + 1, len(sentences), emo_label, sk_emotion,
            effective_speed, sentence[:60],
        )

        try:
            audio_bytes = client.synthesize(
                sentence,
                voice=chosen_voice,
                emotion=sk_emotion if supports_emotion else None,
                speed=effective_speed,
                format=fmt,
            )
        except Exception as exc:
            logger.warning("Sentence %d synthesis failed, skipping: %s", idx, exc)
            continue

        chunk = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        combined += chunk + short_pause

    if len(combined) == 0:
        raise RuntimeError("All sentences failed to synthesize")

    if combined.frame_rate != 48000:
        combined = combined.set_frame_rate(48000)
    if combined.channels != 2:
        combined = combined.set_channels(2)

    target_wav = job_dir / filename
    combined.export(str(target_wav), format="wav")
    logger.info("Created emotional speech WAV: %s (%d sentences)", target_wav, len(sentences))
    return target_wav
