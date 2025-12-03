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

