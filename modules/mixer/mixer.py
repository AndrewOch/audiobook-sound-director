"""
Audio Mixer engine using pydub.

Combines multiple audio tracks with layered volume controls and exports a
final mixed audio file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from .config import MixerConfig, TrackSpec


class AudioMixer:
    """High-level audio mixer using pydub."""

    def __init__(self, config: MixerConfig | None = None):
        self.config = config or MixerConfig()

    # ---------------------
    # Public API
    # ---------------------
    def mix(self, tracks: Iterable[TrackSpec], output_path: str) -> str:
        """
        Mix the provided tracks according to the configuration.

        Args:
            tracks: Iterable of TrackSpec describing input audio files
            output_path: Path to write the mixed output file

        Returns:
            Absolute path to the mixed audio file
        """
        track_list: List[TrackSpec] = list(tracks)
        if not track_list:
            raise ValueError("No tracks provided for mixing")

        segs = [self._load_and_prepare_segment(t) for t in track_list]

        # If any track has a non-zero start_time_s, honour time offsets by
        # overlaying segments onto a silent base with appropriate positions.
        if any(getattr(t, "start_time_s", 0.0) not in (0.0, None) for t in track_list):
            mixed = self._overlay_segments_with_offsets(track_list, segs)
        else:
            mixed = self._overlay_segments(segs)

        if self.config.normalize:
            mixed = self._normalize_peak(mixed, self.config.target_peak_dbfs)

        out_path = Path(output_path).expanduser().resolve()
        if out_path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Output file already exists: {out_path}")

        mixed.export(str(out_path), format=self.config.output_format)
        return str(out_path)

    # ---------------------
    # Internals
    # ---------------------
    def _load_and_prepare_segment(self, track: TrackSpec):
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub is required. Install it with: pip install pydub\n"
                "Also install ffmpeg system package for codec support."
            )

        audio_path = track.resolve_path()
        if not audio_path.exists():
            raise FileNotFoundError(f"Track not found: {audio_path}")

        seg = AudioSegment.from_file(str(audio_path))

        # Resample and set channels
        if seg.frame_rate != self.config.output_sample_rate:
            seg = seg.set_frame_rate(self.config.output_sample_rate)
        if seg.channels != self.config.output_channels:
            seg = seg.set_channels(self.config.output_channels)

        # Layered gains: master + category + background channel + track
        gain_db = float(self.config.master_gain_db)

        if track.kind == 'speech':
            gain_db += float(self.config.speech_gain_db)
        elif track.kind == 'music':
            gain_db += float(self.config.music_gain_db)
        elif track.kind == 'background':
            if track.channel:
                ch_gain = self.config.background_channel_gains_db.get(track.channel, 0.0)
                gain_db += float(ch_gain)

        gain_db += float(track.gain_db)

        if gain_db != 0.0:
            seg = seg.apply_gain(gain_db)

        return seg

    def _overlay_segments(self, segments: List['AudioSegment']):
        assert len(segments) > 0
        base = segments[0]
        for seg in segments[1:]:
            base = base.overlay(seg)
        return base

    def _overlay_segments_with_offsets(
        self,
        tracks: List[TrackSpec],
        segments: List["AudioSegment"],
    ):
        """Overlay segments according to per-track start_time_s offsets."""
        from pydub import AudioSegment

        assert len(tracks) == len(segments) and len(tracks) > 0

        # Compute total duration in milliseconds
        total_ms = 0
        for t, seg in zip(tracks, segments):
            start_s = float(getattr(t, "start_time_s", 0.0) or 0.0)
            if start_s < 0:
                start_s = 0.0
            start_ms = int(start_s * 1000)
            end_ms = start_ms + len(seg)
            if end_ms > total_ms:
                total_ms = end_ms

        if total_ms <= 0:
            # Fallback to simple overlay if something went wrong
            return self._overlay_segments(segments)

        # Create silent base with desired duration/sample rate/channels
        base = AudioSegment.silent(duration=total_ms, frame_rate=self.config.output_sample_rate)
        if base.channels != self.config.output_channels:
            base = base.set_channels(self.config.output_channels)

        # Overlay each segment at its specified position
        for t, seg in zip(tracks, segments):
            start_s = float(getattr(t, "start_time_s", 0.0) or 0.0)
            if start_s < 0:
                start_s = 0.0
            start_ms = int(start_s * 1000)
            base = base.overlay(seg, position=start_ms)

        return base

    def _normalize_peak(self, seg: 'AudioSegment', target_peak_dbfs: float):
        # Shift peak to target by applying gain difference
        shift = target_peak_dbfs - seg.max_dBFS if seg.max_dBFS is not None else 0.0
        if shift != 0.0:
            return seg.apply_gain(shift)
        return seg


