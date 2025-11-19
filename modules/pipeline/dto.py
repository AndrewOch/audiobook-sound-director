"""Dataclass-based DTOs for pipeline inputs, steps, and results."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime
import uuid


InputType = Literal["text", "text_file", "audio_file"]


@dataclass
class JobInfo:
    job_id: str
    job_dir: Path
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "created_at": self.created_at,
        }


@dataclass
class InputRequest:
    input_type: InputType
    text: Optional[str] = None
    text_file_path: Optional[Path] = None
    audio_file_path: Optional[Path] = None
    language: Optional[str] = None


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float


@dataclass
class IngestResult:
    text: str
    audio_path: Optional[Path] = None
    segments: List[TranscriptSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "segments": [asdict(s) for s in self.segments],
            "language": self.language,
            "duration": self.duration,
        }


@dataclass
class StepStatus:
    name: str
    status: Literal["pending", "running", "completed", "skipped", "error"]
    detail: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "artifacts": self._stringify_paths(self.artifacts),
        }

    @staticmethod
    def _stringify_paths(obj: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out


@dataclass
class PipelineRequest:
    request: InputRequest
    job_info: JobInfo


@dataclass
class PipelineResponse:
    job_id: str
    status: Literal["queued", "processing", "completed", "error"]
    steps: Dict[str, StepStatus]
    message: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "message": self.message,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "outputs": self._stringify_paths(self.outputs),
        }

    @staticmethod
    def _stringify_paths(obj: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out


def new_job(output_root: Path) -> JobInfo:
    job_id = str(uuid.uuid4())
    job_dir = output_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return JobInfo(job_id=job_id, job_dir=job_dir, created_at=datetime.now().isoformat())


# -----------------------------
# Track Editor / Mixing DTOs
# -----------------------------

TrackKind = Literal["speech", "music", "background"]


@dataclass
class TrackDTO:
    id: str
    name: str
    kind: TrackKind
    url: str
    path: Path
    channel: Optional[str] = None
    enabled: bool = True
    volume: float = 1.0  # 0.0..1.0 UI volume; server will map to dB

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "url": self.url,
            "path": str(self.path),
            "channel": self.channel,
            "enabled": self.enabled,
            "volume": self.volume,
        }


@dataclass
class MixTrackSetting:
    id: str
    enabled: bool
    volume: float  # 0.0..1.0


@dataclass
class MixRequest:
    job_id: str
    tracks: List[MixTrackSetting]
    # Optional master/category gains could be added later


@dataclass
class MixResponse:
    job_id: str
    status: Literal["ok", "error"]
    download_url: Optional[str] = None
    detail: Optional[str] = None


# -----------------------------
# Segment-based Analysis DTOs
# -----------------------------

@dataclass
class SegmentEmotion:
    """Emotion analysis result for a single segment."""
    segment_id: int
    start: float
    end: float
    text: str
    emotion: str
    confidence: float
    top5: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "top5": self.top5,
        }


@dataclass
class SegmentFoli:
    """Foli classification result for a single segment."""
    segment_id: int
    start: float
    end: float
    text: str
    foli_class: Optional[str] = None
    foli_confidence: Optional[float] = None
    top5: List[Dict[str, Any]] = field(default_factory=list)
    # New: full per-channel predictions (ch1/ch2/ch3)
    channels: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "foli_class": self.foli_class,
            "foli_confidence": self.foli_confidence,
            "top5": self.top5,
            "channels": self.channels,
        }


@dataclass
class ProjectState:
    """Complete project state for saving/loading."""
    job_id: str
    audio_url: str
    duration: float
    segments: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    playhead_position: float = 0.0
    zoom_level: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "audio_url": self.audio_url,
            "duration": self.duration,
            "segments": self.segments,
            "tracks": self.tracks,
            "playhead_position": self.playhead_position,
            "zoom_level": self.zoom_level,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }



