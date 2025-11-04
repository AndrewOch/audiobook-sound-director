"""
Utility functions for Speech Recognition

This module contains helper functions for audio processing,
downloading, and format conversion.
"""

import os
from pathlib import Path
from typing import Union
import tempfile


def download_audio_from_url(url: str, output_path: str = None) -> str:
    """
    Download audio from URL.
    
    Args:
        url: URL to download audio from
        output_path: Path to save audio (optional, creates temp file if None)
    
    Returns:
        Path to downloaded audio file
    
    Raises:
        ImportError: If requests is not installed
        Exception: If download fails
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests library is required for downloading. "
            "Install it with: pip install requests"
        )
    
    if output_path is None:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "audio_download.mp3")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download audio: {e}")


def download_audio_from_drive(drive_url: str, output_path: str = None) -> str:
    """
    Download audio from Google Drive.
    
    Args:
        drive_url: Google Drive URL (either direct or file view link)
        output_path: Path to save audio (optional)
    
    Returns:
        Path to downloaded audio file
    """
    # Extract file ID from various Google Drive URL formats
    if 'file/d/' in drive_url:
        file_id = drive_url.split('file/d/')[1].split('/')[0]
    elif 'id=' in drive_url:
        file_id = drive_url.split('id=')[1].split('&')[0]
    else:
        # Assume it's already a file ID
        file_id = drive_url
    
    # Create direct download URL
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return download_audio_from_url(direct_url, output_path)


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Validate that the file exists and is a supported audio format.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        True if valid, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    # Check file extension
    supported_extensions = {
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus',
        '.webm', '.mp4', '.wma', '.aac'
    }
    
    return file_path.suffix.lower() in supported_extensions


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Duration in seconds
    
    Raises:
        ImportError: If pydub is not installed
        Exception: If file cannot be read
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio duration. "
            "Install it with: pip install pydub"
        )
    
    try:
        audio = AudioSegment.from_file(str(file_path))
        return len(audio) / 1000.0  # Convert ms to seconds
    except Exception as e:
        raise Exception(f"Failed to get audio duration: {e}")


def convert_audio_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: str = 'wav'
) -> str:
    """
    Convert audio file to different format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        output_format: Target format (wav, mp3, etc.)
    
    Returns:
        Path to converted audio file
    
    Raises:
        ImportError: If pydub is not installed
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub is required for audio conversion. "
            "Install it with: pip install pydub\n"
            "Also install ffmpeg system package for format support."
        )
    
    audio = AudioSegment.from_file(str(input_path))
    audio.export(str(output_path), format=output_format)
    
    return str(output_path)


def format_timestamp(seconds: float, include_ms: bool = True) -> str:
    """
    Format timestamp in seconds to HH:MM:SS.mmm or HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        include_ms: Include milliseconds in output
    
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def format_word_timestamps(words: list) -> str:
    """
    Format list of word timestamps into readable text.
    
    Args:
        words: List of word dictionaries with 'text', 'start', 'end'
    
    Returns:
        Formatted string with timestamps
    """
    lines = []
    for word in words:
        start = format_timestamp(word['start'])
        end = format_timestamp(word['end'])
        lines.append(f"[{start} --> {end}] {word['text']}")
    
    return '\n'.join(lines)


def save_transcription_to_file(
    transcription: dict,
    output_path: Union[str, Path],
    format: str = 'txt'
) -> None:
    """
    Save transcription to file.
    
    Args:
        transcription: Transcription dictionary
        output_path: Path to save file
        format: Output format ('txt', 'json', 'srt', 'vtt')
    
    Raises:
        ValueError: If format is not supported
    """
    output_path = Path(output_path)
    
    if format == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription['text'])
    
    elif format == 'json':
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
    
    elif format == 'srt':
        _save_srt(transcription, output_path)
    
    elif format == 'vtt':
        _save_vtt(transcription, output_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_srt(transcription: dict, output_path: Path) -> None:
    """Save transcription in SRT subtitle format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(transcription['segments'], 1):
            start = format_timestamp(segment['start'], include_ms=True).replace('.', ',')
            end = format_timestamp(segment['end'], include_ms=True).replace('.', ',')
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def _save_vtt(transcription: dict, output_path: Path) -> None:
    """Save transcription in WebVTT subtitle format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        
        for segment in transcription['segments']:
            start = format_timestamp(segment['start'], include_ms=True)
            end = format_timestamp(segment['end'], include_ms=True)
            
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text'].strip()}\n\n")

