"""
LLM-based emotional markup for Russian narrative text using Scaleway API.

This module adds vocal/emotional tags (e.g. [sad], [angry], [pause]) into text
before it is passed to TTS. It reuses the same Scaleway LLM setup as
`LLMEmotionClassifier` and is intentionally conservative: on any error it
returns the original text unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .llm_emotion_classifier import LLMEmotionConfig, build_llm_emotion_config_from_env


logger = logging.getLogger("audiobook.emotions.markup")


EMOTIONAL_MARKUP_PROMPT = """You are an expert in adding emotional and vocal annotations to narrative texts.
Your task is to analyze a given Russian text and insert appropriate emotional markup tags to enhance its expressiveness.

Available Emotional Markup Tags:
Use only the tags from this list. Do not invent new tags.
* [angry] - Angry, irritated tone.
* [excited] - Enthusiastic, excited, joyful tone.
* [happy] - Happy, content tone.
* [sad] - Sad, sorrowful tone.
* [anxious] - Nervous, anxious tone.
* [yell] - Conveys a loud, shouting voice.
* [whispers] - Speaks the text in a whisper.
* [sighs] - Inserts the sound of a sigh into speech.
* [laughs] - Inserts laughter into speech.
* [gasp] - Inserts a gasp or sharp intake of breath (from shock).
* [pause] - Inserts a brief pause.
* [long pause] - Inserts a prolonged pause.
* [clears throat] - Inserts the sound of clearing one's throat.

Rules for Applying Markup:
1. Analyze Context: Carefully read the entire text. Understand the situation, character emotions, and dialogue dynamics.
2. Tag Placement: Place the tag immediately before the word, phrase, or sentence where the emotional shift or vocal effect occurs.
3. Scope: A tag applies to all subsequent text until the sentence ends, a new tag is introduced, or for long speeches, after an intervening action or new paragraph.
4. Tone vs. Sound: Distinguish between tone tags ([angry], [sad]) and sound/action tags ([laughs], [gasp], [pause]).
5. Subtlety & Restraint: Use tags purposefully to highlight significant emotional moments. Do not tag every sentence.
6. Preserve Original Text: Do not modify, add, or remove any words from the original Russian text. Only insert the markup tags.

Output Format:
Return only the annotated text in Russian. Do not include explanations, notes, prefixes, markdown formatting or the original plain text."""


class LLMEmotionalMarkup:
    """Client for adding emotional markup using Scaleway LLM."""

    def __init__(self, config: LLMEmotionConfig):
        if OpenAI is None:
            raise ImportError(
                "openai library is required for emotional markup. "
                "Install it with: pip install openai"
            )

        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.logger = logger
        self.logger.info("LLMEmotionalMarkup initialized with model: %s", config.model)

    def _annotate_chunk(self, text: str) -> str:
        """Call LLM for a single chunk of text."""
        if not text.strip():
            return text

        # Optional throttling for markup calls
        try:
            delay_s = float(os.getenv("LLM_MARKUP_REQUEST_DELAY_S", "0") or "0")
        except ValueError:
            delay_s = 0.0
        if delay_s > 0:
            time.sleep(delay_s)

        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": EMOTIONAL_MARKUP_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=min(self.config.max_tokens * 10, 4096),
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            return content or text
        except Exception as exc:
            self.logger.error("LLM markup failed: %s", exc)
            return text

    def annotate_text(self, text: str, max_chunk_chars: int = 4000) -> str:
        """Annotate (possibly long) text with emotional tags using chunking."""
        text = text or ""
        if not text.strip():
            return text

        if len(text) <= max_chunk_chars:
            return self._annotate_chunk(text)

        chunks = _split_text_into_chunks(text, max_chunk_chars=max_chunk_chars)
        annotated_chunks: List[str] = []
        for chunk in chunks:
            annotated_chunks.append(self._annotate_chunk(chunk))
        return "\n\n".join(annotated_chunks)


def _split_text_into_chunks(text: str, max_chunk_chars: int = 4000) -> List[str]:
    """
    Split long text into reasonably sized chunks for markup.

    Preference:
    - split by double newlines (paragraphs)
    - if a paragraph is still huge, split by sentences
    """
    paragraphs = re.split(r"\n\s*\n+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        para = para.rstrip()
        if not para:
            continue
        if len(para) > max_chunk_chars:
            # Flush current and split this huge paragraph by sentences
            flush()
            sentences = re.split(r"(?<=[.!?…])\s+", para)
            cur_sent: List[str] = []
            cur_len = 0
            for s in sentences:
                if not s:
                    continue
                if cur_len + len(s) + 1 > max_chunk_chars and cur_sent:
                    chunks.append(" ".join(cur_sent).strip())
                    cur_sent = [s]
                    cur_len = len(s) + 1
                else:
                    cur_sent.append(s)
                    cur_len += len(s) + 1
            if cur_sent:
                chunks.append(" ".join(cur_sent).strip())
        else:
            if current_len + len(para) + 2 > max_chunk_chars and current:
                flush()
            current.append(para)
            current_len += len(para) + 2

    flush()
    return [c for c in chunks if c]


_markup_client: Optional[LLMEmotionalMarkup] = None


def _get_markup_client() -> Optional[LLMEmotionalMarkup]:
    """Get or create a singleton markup client based on env config."""
    global _markup_client
    if _markup_client is not None:
        return _markup_client

    cfg = build_llm_emotion_config_from_env()
    if cfg is None:
        logger.info(
            "LLMEmotionalMarkup: LLM config not found "
            "(LLM_EMOTIONS_API_KEY / LLM_EMOTIONS_PROJECT_ID). Markup disabled."
        )
        return None
    try:
        _markup_client = LLMEmotionalMarkup(cfg)
        return _markup_client
    except Exception as exc:
        logger.warning("LLMEmotionalMarkup: failed to initialize client: %s", exc)
        _markup_client = None
        return None


def annotate_text(text: str) -> str:
    """
    Public helper: annotate text with emotional tags.

    If LLM config or client is unavailable, returns the original text unchanged.
    """
    client = _get_markup_client()
    if client is None:
        return text or ""
    return client.annotate_text(text)


