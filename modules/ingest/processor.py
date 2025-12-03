"""Input processing for text and audio files.

This module provides `InputProcessor` which accepts various input types
and normalizes them into a single `IngestResult` containing text and
optional audio/transcription metadata saved under the job directory.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil
import logging
import time

from modules.pipeline.dto import (
    InputRequest,
    IngestResult,
    TranscriptSegment,
    JobInfo,
)


class InputProcessor:
    """Process incoming input (text, text file, audio file)."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.logger = logging.getLogger("audiobook.ingest")

    def process(self, request: InputRequest, job: JobInfo) -> IngestResult:
        job_dir = job.job_dir
        job_dir.mkdir(parents=True, exist_ok=True)

        if request.input_type == "text":
            if not request.text:
                raise ValueError("Text input is required for input_type='text'")
            self.logger.info("[Job %s] Ingest: received text input (len=%d)", job.job_id, len(request.text))
            self._save_text(job_dir, request.text)
            
            # Split text into segments for emotion analysis
            segments = self._split_text_into_segments(request.text)
            # Estimate duration: assume ~150 words per minute reading speed
            estimated_duration = self._estimate_duration(request.text)
            self._save_transcript_json(job_dir, request.text, language=None, duration=estimated_duration, segments=segments)
            
            return IngestResult(
                text=request.text,
                segments=segments,
                duration=estimated_duration
            )

        if request.input_type == "text_file":
            if not request.text_file_path or not Path(request.text_file_path).exists():
                raise ValueError("Valid text_file_path is required for input_type='text_file'")
            self.logger.info("[Job %s] Ingest: reading text file %s", job.job_id, request.text_file_path)
            text_content = Path(request.text_file_path).read_text(encoding="utf-8")
            # Also copy original file into job dir for traceability
            dst = job_dir / f"input_{Path(request.text_file_path).name}"
            if Path(request.text_file_path) != dst:
                shutil.copy2(request.text_file_path, dst)
            self._save_text(job_dir, text_content)
            
            # Split text into segments for emotion analysis
            segments = self._split_text_into_segments(text_content)
            # Estimate duration: assume ~150 words per minute reading speed
            estimated_duration = self._estimate_duration(text_content)
            self._save_transcript_json(job_dir, text_content, language=None, duration=estimated_duration, segments=segments)
            
            return IngestResult(
                text=text_content,
                segments=segments,
                duration=estimated_duration
            )

        if request.input_type == "audio_file":
            if not request.audio_file_path or not Path(request.audio_file_path).exists():
                raise ValueError("Valid audio_file_path is required for input_type='audio_file'")
            self.logger.info("[Job %s] Ingest: copying audio file %s", job.job_id, request.audio_file_path)
            # Always copy audio under the job dir to keep all artifacts together
            src_audio = Path(request.audio_file_path)
            job_audio = job_dir / f"input_{src_audio.name}"
            if src_audio != job_audio:
                shutil.copy2(src_audio, job_audio)

            # Create normalized preview wav for clean playback in browser
            preview_wav = self._ensure_preview_wav(job_dir, job_audio)

            # Transcribe using speech recognition module
            try:
                from modules.pipeline.registry import get_speech_recognizer
                self.logger.info("[Job %s] Transcription: using shared recognizer (lang=%s)", job.job_id, request.language or "auto")
                t0 = time.perf_counter()
                recognizer = get_speech_recognizer()
                t1 = time.perf_counter()
                self.logger.info("[Job %s] Transcription: recognizer ready in %.2fs", job.job_id, (t1 - t0))
                self.logger.info("[Job %s] Transcription: started for %s", job.job_id, job_audio)
                t2 = time.perf_counter()
                result = recognizer.transcribe(job_audio, return_word_timestamps=True, language=request.language)
                t3 = time.perf_counter()
                self.logger.info("[Job %s] Transcription: completed in %.2fs", job.job_id, (t3 - t2))

                text = result.get("text", "").strip()
                language = result.get("language")
                duration = result.get("duration")
                segments_dicts = result.get("segments", [])
                segments: List[TranscriptSegment] = []
                for seg in segments_dicts:
                    segments.append(
                        TranscriptSegment(
                            text=seg.get("text", "").strip(),
                            start=float(seg.get("start", 0.0)),
                            end=float(seg.get("end", 0.0)),
                        )
                    )

                # Persist transcript artifacts
                self._save_text(job_dir, text)
                self._save_transcript_json(job_dir, text, language, duration, segments)
                self.logger.info(
                    "[Job %s] Transcription: text_len=%d, segments=%d, lang=%s, duration=%s",
                    job.job_id,
                    len(text),
                    len(segments),
                    language,
                    str(duration),
                )

                return IngestResult(
                    text=text,
                    audio_path=preview_wav if preview_wav else job_audio,
                    segments=segments,
                    language=language,
                    duration=duration,
                )

            except Exception as e:
                # Save basic info and propagate error upwards
                err_path = job_dir / "transcription_error.txt"
                err_path.write_text(str(e), encoding="utf-8")
                self.logger.exception("[Job %s] Transcription failed: %s", job.job_id, e)
                raise

        raise ValueError(f"Unsupported input_type: {request.input_type}")

    @staticmethod
    def _save_text(job_dir: Path, text: str):
        (job_dir / "transcript.txt").write_text(text or "", encoding="utf-8")
        logging.getLogger("audiobook.ingest").info("Saved transcript.txt (%d chars)", len(text or ""))

    @staticmethod
    def _save_transcript_json(
        job_dir: Path,
        text: str,
        language: Optional[str],
        duration: Optional[float],
        segments: List[TranscriptSegment],
    ):
        payload = {
            "text": text,
            "language": language,
            "duration": duration,
            "segments": [asdict(s) for s in segments],
        }
        with open(job_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _split_text_into_segments(self, text: str) -> List[TranscriptSegment]:
        """
        Split text into segments for emotion analysis.
        Splits by sentences (periods, exclamation, question marks).
        """
        import re
        
        # Split by sentence endings (., !, ?, followed by space or newline)
        # Keep the punctuation with the sentence
        sentence_pattern = r'([^.!?\n]+[.!?]+)\s*'
        sentences = re.findall(sentence_pattern, text)
        
        # If no sentences found, split by newlines or use whole text
        if not sentences:
            # Try splitting by newlines
            parts = [p.strip() for p in text.split('\n') if p.strip()]
            if not parts:
                parts = [text.strip()] if text.strip() else []
            sentences = parts
        
        segments: List[TranscriptSegment] = []
        current_time = 0.0
        
        # Estimate time per word: ~150 words per minute = 0.4 seconds per word
        words_per_second = 150 / 60  # ~2.5 words per second
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Estimate duration based on word count
            word_count = len(sentence.split())
            duration = word_count / words_per_second if word_count > 0 else 1.0
            # Minimum segment duration: 0.5 seconds
            duration = max(0.5, duration)
            
            segments.append(
                TranscriptSegment(
                    text=sentence,
                    start=current_time,
                    end=current_time + duration
                )
            )
            current_time += duration
        
        return segments
    
    def _estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration for text based on reading speed.
        Assumes ~150 words per minute reading speed.
        """
        word_count = len(text.split())
        words_per_minute = 150
        duration_minutes = word_count / words_per_minute if word_count > 0 else 0.0
        return duration_minutes * 60  # Convert to seconds

    def _ensure_preview_wav(self, job_dir: Path, src_audio: Path) -> Optional[Path]:
        try:
            from pydub import AudioSegment
        except Exception:
            self.logger.warning("pydub not available; skipping preview wav normalization")
            return None
        try:
            dst = job_dir / "speech.wav"
            if dst.exists():
                return dst
            seg = AudioSegment.from_file(str(src_audio))
            if seg.frame_rate != 48000:
                seg = seg.set_frame_rate(48000)
            if seg.channels != 2:
                seg = seg.set_channels(2)
            seg.export(str(dst), format="wav")
            self.logger.info("Created normalized preview wav: %s", dst)
            return dst
        except Exception as e:
            self.logger.warning("Failed to create preview wav for %s: %s", src_audio, e)
            return None


