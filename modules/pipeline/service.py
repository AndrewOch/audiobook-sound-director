"""Central pipeline service orchestrating modules.

The PipelineService plans and/or executes the audiobook pipeline steps,
starting from input ingestion, through optional analysis/generation
modules, and ending with mixing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import os
import logging
import time
import importlib
from datetime import datetime
import re

from .dto import (
    InputRequest,
    PipelineRequest,
    PipelineResponse,
    StepStatus,
    IngestResult,
    JobInfo,
    new_job,
    SegmentEmotion,
    SegmentFoli,
)
from modules.ingest import InputProcessor
from modules.pipeline.registry import (
    get_emotion_classifier,
    get_foli_classifier,
    get_music_generator,
)


@dataclass
class PipelineServiceConfig:
    enable_emotions: bool = True
    enable_foli_classification: bool = True
    enable_music_generation: bool = False
    enable_speech_generation: bool = False
    enable_foli_generation: bool = False
    enable_mixing: bool = False
    save_intermediate: bool = True


class PipelineService:
    """Orchestrates the audiobook pipeline across modules."""

    def __init__(self, output_root: Optional[Path] = None, config: Optional[PipelineServiceConfig] = None):
        # Infer project root based on this file location: modules/pipeline/service.py
        self._this_dir = Path(__file__).resolve()
        self._project_root = self._this_dir.parents[2]
        self._output_root = Path(output_root) if output_root else (self._project_root / "output")
        self._output_root.mkdir(parents=True, exist_ok=True)
        self.config = config or PipelineServiceConfig()
        self.ingest_processor = InputProcessor(self._output_root)
        self.logger = logging.getLogger("audiobook.pipeline")

    def start_job(self, input_request: InputRequest, job: Optional[JobInfo] = None, execute: bool = False) -> PipelineResponse:
        """Create/accept a job, run ingestion, and optionally execute steps.

        Args:
            input_request: InputRequest DTO
            job: Optional existing JobInfo (e.g., created by a web layer)
            execute: If True, immediately run all enabled steps synchronously
        """
        job_info = job or new_job(self._output_root)

        # Initialize step statuses
        steps: Dict[str, StepStatus] = {
            "ingest": StepStatus(name="ingest", status="running"),
            "transcription": StepStatus(name="transcription", status="pending"),
            "emotion_analysis": StepStatus(name="emotion_analysis", status="pending"),
            "foli_classification": StepStatus(name="foli_classification", status="pending"),
            "speech_generation": StepStatus(name="speech_generation", status="pending"),
            "music_generation": StepStatus(name="music_generation", status="pending"),
            "foli_generation": StepStatus(name="foli_generation", status="pending"),
            "mixing": StepStatus(name="mixing", status="pending"),
        }
        self._save_status(job_info, steps, status="processing", message="Job started - ingestion running")

        # Ingest
        try:
            self.logger.info("[Job %s] Ingestion started (type=%s, dir=%s)", job_info.job_id, input_request.input_type, job_info.job_dir)
            t0 = time.perf_counter()
            ingest_result = self.ingest_processor.process(input_request, job_info)
            t1 = time.perf_counter()
            self.logger.info(
                "[Job %s] Ingestion completed in %.2fs (text_len=%d, audio=%s)",
                job_info.job_id,
                (t1 - t0),
                len(ingest_result.text or ""),
                str(ingest_result.audio_path) if ingest_result.audio_path else "none",
            )
            steps["ingest"].status = "completed"
            steps["ingest"].artifacts = {"job_dir": job_info.job_dir}
            # Mark transcription step depending on input type
            if input_request.input_type == "audio_file":
                steps["transcription"].status = "completed"
                steps["transcription"].artifacts = {
                    "transcript.txt": job_info.job_dir / "transcript.txt",
                    "transcript.json": job_info.job_dir / "transcript.json",
                }
            else:
                steps["transcription"].status = "skipped"
            self._save_status(job_info, steps, status="processing", message="Ingestion completed")
        except Exception as e:
            self.logger.exception("[Job %s] Ingestion failed: %s", job_info.job_id, e)
            steps["ingest"].status = "error"
            steps["ingest"].detail = str(e)
            self._save_status(job_info, steps, status="error", message="Ingestion failed")
            return PipelineResponse(
                job_id=job_info.job_id,
                status="error",
                steps=steps,
                message="Ingestion failed",
            )

        if not execute:
            # Return plan and basic ingestion outputs without running heavy models
            self._mark_planned_steps(steps)
            tracks = self._build_tracks(job_info, ingest_result)
            self._save_status(job_info, steps, status="processing", message="Planned steps after ingestion", outputs={
                "tracks": [t.to_dict() for t in tracks]
            })
            return PipelineResponse(
                job_id=job_info.job_id,
                status="processing",
                steps=steps,
                message="Job created and input ingested. Steps planned.",
                outputs={
                    "job_dir": job_info.job_dir,
                    "text_preview": ingest_result.text[:200] if ingest_result.text else None,
                    "tracks": [t.to_dict() for t in tracks],
                },
            )

        # Execute enabled steps synchronously (may take time)
        try:
            emotions = None
            speech_path = None
            if self.config.enable_emotions:
                self.logger.info("[Job %s] Emotion analysis started", job_info.job_id)
                steps["emotion_analysis"].status = "running"
                self._save_status(job_info, steps, status="processing", message="Emotion analysis running")
                t0 = time.perf_counter()
                emotions, emotion_artifacts = self._run_emotion_analysis(ingest_result, job_info)
                t1 = time.perf_counter()
                self.logger.info("[Job %s] Emotion analysis completed in %.2fs", job_info.job_id, (t1 - t0))
                steps["emotion_analysis"].status = "completed"
                steps["emotion_analysis"].artifacts = emotion_artifacts
                self._save_status(job_info, steps, status="processing", message="Emotion analysis completed")
            else:
                steps["emotion_analysis"].status = "skipped"

            # Speech generation (for text/text_file inputs without original audio)
            if self.config.enable_speech_generation and ingest_result.audio_path is None:
                self.logger.info("[Job %s] Speech generation started", job_info.job_id)
                steps["speech_generation"].status = "running"
                self._save_status(
                    job_info,
                    steps,
                    status="processing",
                    message="Speech generation running",
                )
                t0 = time.perf_counter()
                speech_path, speech_artifacts = self._run_speech_generation(ingest_result, job_info)
                t1 = time.perf_counter()
                self.logger.info(
                    "[Job %s] Speech generation completed in %.2fs -> %s",
                    job_info.job_id,
                    (t1 - t0),
                    speech_path,
                )
                steps["speech_generation"].status = "completed"
                steps["speech_generation"].artifacts = speech_artifacts
                self._save_status(
                    job_info,
                    steps,
                    status="processing",
                    message="Speech generation completed",
                )
            else:
                # For audio_file inputs or when disabled we skip speech generation
                steps["speech_generation"].status = "skipped"

            if self.config.enable_foli_classification:
                self.logger.info("[Job %s] Foli classification started", job_info.job_id)
                steps["foli_classification"].status = "running"
                self._save_status(job_info, steps, status="processing", message="Foli classification running")
                t0 = time.perf_counter()
                foli_preds, foli_artifacts = self._run_foli_classification(ingest_result, job_info)
                t1 = time.perf_counter()
                self.logger.info("[Job %s] Foli classification completed in %.2fs", job_info.job_id, (t1 - t0))
                steps["foli_classification"].status = "completed"
                steps["foli_classification"].artifacts = foli_artifacts
                self._save_status(job_info, steps, status="processing", message="Foli classification completed")
            else:
                steps["foli_classification"].status = "skipped"

            if self.config.enable_music_generation:
                self.logger.info("[Job %s] Music generation started", job_info.job_id)
                steps["music_generation"].status = "running"
                self._save_status(job_info, steps, status="processing", message="Music generation running")
                t0 = time.perf_counter()
                music_path, music_artifacts = self._run_music_generation(emotions, job_info)
                t1 = time.perf_counter()
                self.logger.info("[Job %s] Music generation completed in %.2fs -> %s", job_info.job_id, (t1 - t0), music_path)
                steps["music_generation"].status = "completed"
                steps["music_generation"].artifacts = music_artifacts
                self._save_status(job_info, steps, status="processing", message="Music generation completed")
            else:
                steps["music_generation"].status = "skipped"

            # Foli background generation (optional, attempt dynamic import)
            if self.config.enable_foli_generation:
                try:
                    self.logger.info("[Job %s] Foli generation started", job_info.job_id)
                    self.logger.info("[Job %s] ВНИМАНИЕ: При первом запуске загружается модель AudioLDM2 (~2-3 ГБ, 26 файлов). Это может занять 5-10 минут.", job_info.job_id)
                    steps["foli_generation"].status = "running"
                    self._save_status(job_info, steps, status="processing", message="Foli generation running - загрузка модели AudioLDM2 (может занять несколько минут при первом запуске)")
                    t0 = time.perf_counter()
                    foli_paths, foli_artifacts = self._run_foli_generation(job_info)
                    t1 = time.perf_counter()
                    self.logger.info("[Job %s] Foli generation completed in %.2fs -> %d tracks", job_info.job_id, (t1 - t0), len(foli_paths))
                    steps["foli_generation"].status = "completed"
                    steps["foli_generation"].artifacts = foli_artifacts
                    self._save_status(job_info, steps, status="processing", message="Foli generation completed")
                except Exception as e:
                    self.logger.exception("[Job %s] Foli generation failed: %s", job_info.job_id, e)
                    steps["foli_generation"].status = "error"
                    steps["foli_generation"].detail = str(e)
                    self._save_status(job_info, steps, status="processing", message=f"Foli generation failed: {e}")
            else:
                steps["foli_generation"].status = "skipped"

            # Placeholder for not-yet-implemented mixing module
            steps["mixing"].status = "skipped" if not self.config.enable_mixing else "pending"

            tracks = self._build_tracks(job_info, ingest_result)
            # Save track list descriptor for frontend reuse
            with open(job_info.job_dir / "tracks.json", "w", encoding="utf-8") as f:
                json.dump([t.to_dict() for t in tracks], f, ensure_ascii=False, indent=2)
            self.logger.info("[Job %s] Built %d track descriptors (tracks.json)", job_info.job_id, len(tracks))
            self._save_status(job_info, steps, status="completed", message="Pipeline executed successfully", outputs={
                "tracks": [t.to_dict() for t in tracks]
            })

            return PipelineResponse(
                job_id=job_info.job_id,
                status="completed",
                steps=steps,
                message="Pipeline executed successfully",
                outputs={
                    "job_dir": job_info.job_dir,
                    "music": music_path if self.config.enable_music_generation else None,
                    "tracks": [t.to_dict() for t in tracks],
                },
            )

        except Exception as e:
            # Mark the last running step as error if possible
            for step in steps.values():
                if step.status == "running":
                    step.status = "error"
                    step.detail = str(e)
                    break
            self.logger.exception("[Job %s] Pipeline execution failed: %s", job_info.job_id, e)
            self._save_status(job_info, steps, status="error", message=str(e))
            return PipelineResponse(
                job_id=job_info.job_id,
                status="error",
                steps=steps,
                message=str(e),
            )

    # -----------------------------
    # Segment utilities (merge for classification context)
    # -----------------------------
    @staticmethod
    def _ends_with_sentence_terminator(text: str) -> bool:
        if not text:
            return False
        t = str(text).strip()
        # Allow trailing quotes/brackets after terminator
        return bool(re.search(r'[.!?…]+["»\')\]]*$', t))

    def _merge_transcript_segments(self, segments: List["TranscriptSegment"], max_gap_s: float = 0.5):
        """
        Merge adjacent Whisper segments for richer context if:
        - previous segment does NOT end with sentence terminator
        - and the gap to the next segment is less than max_gap_s.
        Returns a list of merged groups with mapping to original indices.
        """
        merged = []
        if not segments:
            return merged
        # Initialize first group
        current = {
            "original_ids": [0],
            "start": float(segments[0].start),
            "end": float(segments[0].end),
            "text": (segments[0].text or "").strip(),
        }
        for i in range(len(segments) - 1):
            cur_seg = segments[i]
            nxt_seg = segments[i + 1]
            gap = float(nxt_seg.start) - float(cur_seg.end)
            # Merge condition
            if (not self._ends_with_sentence_terminator(cur_seg.text)) and (gap < max_gap_s):
                current["original_ids"].append(i + 1)
                current["end"] = float(nxt_seg.end)
                # Concatenate with space
                nxt_text = (nxt_seg.text or "").strip()
                if nxt_text:
                    if current["text"]:
                        current["text"] += " " + nxt_text
                    else:
                        current["text"] = nxt_text
            else:
                merged.append(current)
                current = {
                    "original_ids": [i + 1],
                    "start": float(nxt_seg.start),
                    "end": float(nxt_seg.end),
                    "text": (nxt_seg.text or "").strip(),
                }
        merged.append(current)
        return merged
    
    def _write_transcript_merged(self, job: JobInfo, merged_groups: List[Dict], total_duration: Optional[float] = None) -> None:
        """Persist merged transcript to transcript_merged.json for UI."""
        try:
            if total_duration is None:
                try:
                    total_duration = max([float(g.get("end", 0.0)) for g in merged_groups]) if merged_groups else 0.0
                except Exception:
                    total_duration = 0.0
            payload = {
                "segments": [
                    {
                        "text": g.get("text", ""),
                        "start": float(g.get("start", 0.0)),
                        "end": float(g.get("end", 0.0)),
                        "original_ids": g.get("original_ids", []),
                    }
                    for g in merged_groups
                ]
            }
            # Include total duration for timeline
            payload["duration"] = float(total_duration or 0.0)
            with open(job.job_dir / "transcript_merged.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("[Job %s] Failed to write transcript_merged.json: %s", job.job_id, e)

    def _mark_planned_steps(self, steps: Dict[str, StepStatus]):
        # Emotions
        steps["emotion_analysis"].status = "pending" if self.config.enable_emotions else "skipped"
        # Foli classification
        steps["foli_classification"].status = "pending" if self.config.enable_foli_classification else "skipped"
        # Music generation
        steps["music_generation"].status = "pending" if self.config.enable_music_generation else "skipped"
        # Others
        steps["speech_generation"].status = "pending" if self.config.enable_speech_generation else "skipped"
        steps["foli_generation"].status = "pending" if self.config.enable_foli_generation else "skipped"
        steps["mixing"].status = "pending" if self.config.enable_mixing else "skipped"

    def _run_emotion_analysis(self, ingest_result: IngestResult, job: JobInfo) -> Tuple[List[Tuple[str, float]], Dict[str, Path]]:
        clf = get_emotion_classifier()
        
        # Analyze overall text for backward compatibility
        pred = clf.predict(ingest_result.text)
        top5 = pred.get("top5", [])
        # Convert to (emotion, prob) tuples
        emotions: List[Tuple[str, float]] = [(item["emotion"], float(item["prob"])) for item in top5]

        # Save overall emotions JSON
        out_path = job.job_dir / "emotions.json"
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)

        # Analyze segments separately with context merging (by merged group index)
        segments_emotions: List[SegmentEmotion] = []
        if ingest_result.segments:
            merged_groups = self._merge_transcript_segments(ingest_result.segments, max_gap_s=0.5)
            # Save merged transcript for UI (with duration)
            self._write_transcript_merged(job, merged_groups, total_duration=ingest_result.duration)
            self.logger.info(
                "[Job %s] Emotion analysis: original segments=%d, merged groups=%d",
                job.job_id, len(ingest_result.segments), len(merged_groups)
            )
            for group_idx, group in enumerate(merged_groups):
                text = (group.get("text") or "").strip()
                if not text:
                    continue
                seg_pred = clf.predict(text)
                seg_top5 = seg_pred.get("top5", [])
                emotion_label = seg_pred.get("emotion", "neutral")
                confidence = seg_pred.get("confidence", 0.0)
                # Use merged group index and boundaries
                segments_emotions.append(SegmentEmotion(
                    segment_id=group_idx,
                    start=float(group.get("start", 0.0)),
                    end=float(group.get("end", 0.0)),
                    text=text,
                    emotion=emotion_label,
                    confidence=confidence,
                    top5=seg_top5,
                ))
            
            # Save segment-based emotions
            segments_path = job.job_dir / "segments_emotions.json"
            with open(segments_path, "w", encoding="utf-8") as f:
                json.dump([seg.to_dict() for seg in segments_emotions], f, ensure_ascii=False, indent=2)
            self.logger.info("[Job %s] Saved emotions for %d segments", job.job_id, len(segments_emotions))

        return emotions, {"emotions.json": out_path, "segments_emotions.json": segments_path if segments_emotions else None}

    def _run_foli_classification(self, ingest_result: IngestResult, job: JobInfo) -> Tuple[Dict, Dict[str, Path]]:
        clf = get_foli_classifier()
        backend = "onnx" if clf.__class__.__name__.endswith("ONNX") else "pytorch"
        
        # Classify overall text for backward compatibility
        pred = clf.predict(ingest_result.text)
        self.logger.info("[Job %s] Foli classification backend: %s", job.job_id, backend)

        # Save overall predictions
        out_path = job.job_dir / "foli_predictions.json"
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)

        # Classify segments separately with context merging (by merged group index)
        segments_foli: List[SegmentFoli] = []
        if ingest_result.segments:
            merged_groups = self._merge_transcript_segments(ingest_result.segments, max_gap_s=0.5)
            # Ensure merged transcript persisted (if not already), include duration
            self._write_transcript_merged(job, merged_groups, total_duration=ingest_result.duration)
            self.logger.info(
                "[Job %s] Foli classification: original segments=%d, merged groups=%d",
                job.job_id, len(ingest_result.segments), len(merged_groups)
            )
            for group_idx, group in enumerate(merged_groups):
                text = (group.get("text") or "").strip()
                if not text:
                    continue
                seg_pred = clf.predict(text)
                # seg_pred is expected to be channels dict {'ch1': {...}, 'ch2': {...}, 'ch3': {...}}
                channels = seg_pred if isinstance(seg_pred, dict) else {}
                # Apply confidence threshold: if prob < 0.4 -> Silence
                norm_channels = {}
                threshold = 0.4
                for ch in ("ch1", "ch2", "ch3"):
                    ch_data = dict((channels.get(ch) or {}))
                    prob = float(ch_data.get("prob") or ch_data.get("score") or 0.0)
                    klass = ch_data.get("class") or ch_data.get("label")
                    if prob < threshold or not klass:
                        norm_channels[ch] = {
                            "class": "Silence",
                            "prob": prob,
                            "top5": ch_data.get("top5", []),
                        }
                    else:
                        norm_channels[ch] = {
                            "class": klass,
                            "prob": prob,
                            "top5": ch_data.get("top5", []),
                        }
                ch1 = norm_channels.get("ch1") or {}
                segments_foli.append(SegmentFoli(
                    segment_id=group_idx,
                    start=float(group.get("start", 0.0)),
                    end=float(group.get("end", 0.0)),
                    text=text,
                    foli_class=ch1.get("class"),
                    foli_confidence=ch1.get("prob") or ch1.get("score"),
                    top5=ch1.get("top5", []),
                    channels=norm_channels,
                ))
            
            # Save segment-based foli classifications
            segments_path = job.job_dir / "segments_foli.json"
            with open(segments_path, "w", encoding="utf-8") as f:
                json.dump([seg.to_dict() for seg in segments_foli], f, ensure_ascii=False, indent=2)
            self.logger.info("[Job %s] Saved foli classifications for %d segments", job.job_id, len(segments_foli))

        return pred, {"foli_predictions.json": out_path, "segments_foli.json": segments_path if segments_foli else None}

    def _run_music_generation(self, emotions: Optional[List[Tuple[str, float]]], job: JobInfo) -> Tuple[Path, Dict[str, Path]]:
        generator = get_music_generator()
        # Use emotions if provided, else a safe default prompt
        if emotions:
            self.logger.info("[Job %s] Music prompt from emotions: %s", job.job_id, ", ".join([e for e, _ in emotions]))
            audio = generator.generate_from_emotions(emotions, duration_seconds=22)
        else:
            audio = generator.generate_from_prompt("Cinematic ambient background for audiobook narration", duration_seconds=22)

        music_path = job.job_dir / "music.wav"
        generator.save_audio(audio, music_path)

        return music_path, {"music.wav": music_path}

    def _run_speech_generation(self, ingest_result: IngestResult, job: JobInfo) -> Tuple[Path, Dict[str, Path]]:
        """Generate speech audio from text using ElevenLabs with emotional markup."""
        from modules.speech import generate_speech_with_emotions

        text = (ingest_result.text or "").strip()
        if not text:
            raise ValueError("Cannot generate speech for empty text")

        speech_path = generate_speech_with_emotions(text=text, job_dir=job.job_dir)
        return speech_path, {"speech.wav": speech_path}

    def _run_foli_generation(self, job: JobInfo) -> Tuple[List[Path], Dict[str, Path]]:
        """Generate background foley tracks via explicit module import.

        Uses `modules.foli_generation.FoliGenerator` just like other modules
        (no reflection). Prompts are derived from saved `foli_predictions.json`.
        """
        from modules.foli_generation import FoliGenerator, FoliGenConfig

        preds_path = job.job_dir / "foli_predictions.json"
        preds = None
        if preds_path.exists():
            try:
                with open(preds_path, "r", encoding="utf-8") as f:
                    preds = json.load(f)
            except Exception as e:
                self.logger.warning("[Job %s] Failed to read foli_predictions.json: %s", job.job_id, e)
                preds = None

        def label_to_prompt(label: str) -> str:
            clean = str(label).replace('_', ' ').lower()
            return f"The sound of {clean}. High quality, clear."

        def build_prompt(ch_pred: Dict | None) -> str:
            if not isinstance(ch_pred, dict):
                return "Ambient background noise, high quality, clear."
            top = ch_pred.get('class') or (ch_pred.get('top5', [{}])[0].get('class') if ch_pred.get('top5') else None)
            if not top:
                return "Ambient background noise, high quality, clear."
            return label_to_prompt(top)

        self.logger.info("[Job %s] Инициализация генератора фоли звуков...", job.job_id)
        generator = FoliGenerator(FoliGenConfig())
        
        # Предзагружаем модель, чтобы пользователь видел прогресс загрузки
        self.logger.info("[Job %s] Загрузка модели AudioLDM2 (это может занять несколько минут при первом запуске)...", job.job_id)
        generator.load_model()
        self.logger.info("[Job %s] Модель AudioLDM2 загружена, начинаем генерацию фоли звуков...", job.job_id)

        paths: List[Path] = []
        artifacts: Dict[str, Path] = {}
        seeds = {"ch1": 0, "ch2": 1, "ch3": 2}

        for ch in ("ch1", "ch2", "ch3"):
            self.logger.info("[Job %s] Генерация фоли звука для канала %s...", job.job_id, ch)
            prompt = build_prompt(preds.get(ch) if isinstance(preds, dict) else None)
            audio = generator.generate(
                prompt=prompt,
                audio_length_in_s=generator.config.audio_length_in_s,
                negative_prompt=generator.config.negative_prompt or "Low quality.",
                num_inference_steps=generator.config.num_inference_steps,
                num_waveforms_per_prompt=1,
                seed=seeds[ch],
            )
            out_path = job.job_dir / f"foli_{ch}.wav"
            generator.save_audio(audio, out_path)
            paths.append(out_path)
            artifacts[f"foli_{ch}.wav"] = out_path

        return paths, artifacts

    def _normalize_foli_result(self, result):
        if result is None:
            return None
        paths: List[Path] = []
        artifacts: Dict[str, Path] = {}
        if isinstance(result, dict):
            for ch, p in result.items():
                p = Path(p)
                paths.append(p)
                artifacts[f"foli_{ch}.wav"] = p
            return paths, artifacts
        if isinstance(result, list):
            chs = ("ch1", "ch2", "ch3")
            for idx, p in enumerate(result):
                p = Path(p)
                ch = chs[idx] if idx < len(chs) else f"ch{idx+1}"
                paths.append(p)
                artifacts[f"foli_{ch}.wav"] = p
            return paths, artifacts
        return None

    def _build_tracks(self, job: JobInfo, ingest_result: IngestResult):
        from .dto import TrackDTO

        tracks: List[TrackDTO] = []

        # Speech track from normalized preview wav if exists, else original audio
        speech_wav = job.job_dir / "speech.wav"
        speech_path = speech_wav if speech_wav.exists() else ingest_result.audio_path
        if speech_path and speech_path.exists():
            url = self._to_web_url(speech_path)
            tracks.append(TrackDTO(
                id="speech",
                name="Речь",
                kind="speech",
                url=url,
                path=speech_path,
                enabled=True,
                volume=1.0,
            ))

        # Music track
        music_path = job.job_dir / "music.wav"
        if music_path.exists():
            tracks.append(TrackDTO(
                id="music",
                name="Музыка",
                kind="music",
                url=self._to_web_url(music_path),
                path=music_path,
                enabled=True,
                volume=0.35,
            ))

        # Foli background channels if present
        for ch in ("ch1", "ch2", "ch3"):
            p = job.job_dir / f"foli_{ch}.wav"
            if p.exists():
                tracks.append(TrackDTO(
                    id=f"foli_{ch}",
                    name=f"Фон {ch}",
                    kind="background",
                    channel=ch,
                    url=self._to_web_url(p),
                    path=p,
                    enabled=True,
                    volume=0.6,
                ))

        self.logger.info("[Job %s] Tracks collected: %s", job.job_id, ", ".join([t.id for t in tracks]))
        return tracks

    def _to_web_url(self, path: Path) -> str:
        # Convert absolute path within output root to /output/<job>/<file>
        path = Path(path)
        # Expect structure: <root>/output/<job>/<file>
        parts = list(path.parts)
        if "output" in parts:
            idx = parts.index("output")
            rel = Path(*parts[idx:])
            return "/" + str(rel).replace(os.sep, "/")
        # Fallback: return file name relative to job
        return f"/output/{path.name}"

    def _save_status(self, job: JobInfo, steps: Dict[str, StepStatus], status: str, message: Optional[str] = None, outputs: Optional[Dict] = None):
        try:
            data = {
                "job_id": job.job_id,
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "steps": {k: v.to_dict() for k, v in steps.items()},
                "outputs": outputs or {},
            }
            # Stringify Paths in outputs
            for k, v in list(data["outputs"].items()):
                if isinstance(v, Path):
                    data["outputs"][k] = str(v)
                if isinstance(v, list):
                    data["outputs"][k] = [
                        (str(x) if isinstance(x, Path) else x) for x in v
                    ]
            with open(job.job_dir / "job_status.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("[Job %s] Failed to write job_status.json: %s", job.job_id, e)


