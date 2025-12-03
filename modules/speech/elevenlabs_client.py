"""
Simple ElevenLabs text-to-speech client.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import os
import uuid

import requests

from modules.emotions.llm_emotional_markup import annotate_text


logger = logging.getLogger("audiobook.speech.elevenlabs")


@dataclass
class ElevenLabsConfig:
    api_key: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default female voice
    model_id: str = "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    base_url: str = "https://api.elevenlabs.io"


class ElevenLabsClient:
    def __init__(self, config: ElevenLabsConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": "audio/mpeg",
                "xi-api-key": config.api_key,
                "Content-Type": "application/json",
            }
        )

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> bytes:
        """
        Generate speech audio bytes for the given text.
        """
        payload = {
            "text": text,
            "model_id": model_id or self.config.model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "use_speaker_boost": self.config.use_speaker_boost,
            },
        }
        vid = voice_id or self.config.voice_id
        url = f"{self.config.base_url}/v1/text-to-speech/{vid}"
        resp = self.session.post(url, json=payload, timeout=120)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("ElevenLabs synthesis failed: %s", resp.text)
            raise exc
        return resp.content

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_bytes = self.synthesize(text=text, voice_id=voice_id, model_id=model_id)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path


_client: Optional[ElevenLabsClient] = None


def get_elevenlabs_client() -> ElevenLabsClient:
    global _client
    if _client:
        return _client

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY environment variable is not set")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", ElevenLabsConfig(api_key=api_key).voice_id)
    model_id = os.getenv("ELEVENLABS_MODEL_ID", ElevenLabsConfig(api_key=api_key).model_id)

    config = ElevenLabsConfig(
        api_key=api_key,
        voice_id=voice_id,
        model_id=model_id,
    )
    _client = ElevenLabsClient(config)
    return _client


def safe_filename(prefix: str = "speech", suffix: str = ".mp3") -> str:
    return f"{prefix}_{uuid.uuid4().hex}{suffix}"


def generate_speech_with_emotions(
    text: str,
    job_dir: Path,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    filename: str = "speech.wav",
) -> Path:
    """
    Generate speech audio for the given text with LLM-based emotional markup.

    Steps:
    - annotate text with emotional/vocal tags via Scaleway LLM (if configured);
    - synthesize speech via ElevenLabs TTS;
    - convert to normalized WAV (48 kHz, stereo) saved as `filename` in job_dir.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    raw_text = text or ""
    marked_text = annotate_text(raw_text)

    client = get_elevenlabs_client()

    # First synthesize to MP3 (as ElevenLabs returns audio/mpeg by default)
    tmp_mp3 = job_dir / safe_filename(prefix="speech_raw", suffix=".mp3")
    logger.info("Generating ElevenLabs speech with emotional markup -> %s", tmp_mp3)
    client.synthesize_to_file(
        text=marked_text,
        output_path=tmp_mp3,
        voice_id=voice_id,
        model_id=model_id,
    )

    # Then convert to normalized WAV for consistent playback in UI
    target_wav = job_dir / filename
    try:
        from pydub import AudioSegment  # type: ignore

        seg = AudioSegment.from_file(str(tmp_mp3))
        if seg.frame_rate != 48000:
            seg = seg.set_frame_rate(48000)
        if seg.channels != 2:
            seg = seg.set_channels(2)
        seg.export(str(target_wav), format="wav")
        logger.info("Created normalized speech WAV: %s", target_wav)
        return target_wav
    except Exception as exc:
        logger.warning(
            "Failed to convert ElevenLabs MP3 to WAV, using raw MP3 instead: %s", exc
        )
        return tmp_mp3


