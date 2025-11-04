"""
Speech Recognizer

This module provides the main interface for speech recognition
using Whisper model with word-level timestamps.
"""

from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from .config import RecognizerConfig, DEFAULT_CONFIG
from .model import WhisperModel
from .utils import (
    download_audio_from_url,
    download_audio_from_drive,
    validate_audio_file,
    get_audio_duration,
)


class SpeechRecognizer:
    """
    Main speech recognition interface.
    
    Provides methods for transcribing audio files with word-level
    timestamps using OpenAI's Whisper model.
    """
    
    def __init__(self, config: RecognizerConfig = None):
        """
        Initialize speech recognizer.
        
        Args:
            config: Recognizer configuration. If None, uses default config.
        """
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        self.model = WhisperModel(
            model_size=self.config.model_size,
            device=self.config.device,
            download_root=self.config.download_root
        )
    
    def transcribe(
        self,
        audio_source: Union[str, Path],
        return_word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with timestamps.
        
        Args:
            audio_source: Path to audio file, URL, or Google Drive link
            return_word_timestamps: Include word-level timestamps in output
        
        Returns:
            Dictionary with transcription:
            {
                'text': str,  # Full transcription text
                'segments': [  # Segment-level results
                    {
                        'text': str,
                        'start': float,
                        'end': float,
                        'words': [...]  # If word_timestamps=True
                    },
                    ...
                ],
                'language': str,  # Detected or specified language
                'duration': float,  # Audio duration in seconds
            }
        """
        # Handle different audio sources
        audio_path = self._prepare_audio_source(audio_source)
        
        # Validate audio file
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")
        
        # Get audio duration
        try:
            duration = get_audio_duration(audio_path)
        except:
            duration = None
        
        # Transcribe using Whisper
        result = self.model.transcribe(
            audio_path,
            language=self.config.language,
            task=self.config.task,
            word_timestamps=self.config.word_timestamps,
            temperature=self.config.temperature,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            patience=self.config.patience,
            verbose=self.config.verbose,
        )
        
        # Format output
        output = {
            'text': result.get('text', '').strip(),
            'segments': result.get('segments', []),
            'language': result.get('language', self.config.language),
            'duration': duration,
        }
        
        # Remove word timestamps if not requested
        if not return_word_timestamps:
            for segment in output['segments']:
                if 'words' in segment:
                    del segment['words']
        
        return output
    
    def transcribe_segments(
        self,
        audio_source: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio and return only segments.
        
        Args:
            audio_source: Path to audio file, URL, or Google Drive link
        
        Returns:
            List of segments with text and timestamps
        """
        result = self.transcribe(audio_source, return_word_timestamps=False)
        return result['segments']
    
    def transcribe_words(
        self,
        audio_source: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio and return word-level timestamps.
        
        Args:
            audio_source: Path to audio file, URL, or Google Drive link
        
        Returns:
            List of words with text and timestamps:
            [
                {'text': str, 'start': float, 'end': float},
                ...
            ]
        """
        result = self.transcribe(audio_source, return_word_timestamps=True)
        
        # Extract all words from all segments
        words = []
        for segment in result['segments']:
            if 'words' in segment:
                for word_info in segment['words']:
                    words.append({
                        'text': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0.0),
                        'end': word_info.get('end', 0.0),
                    })
        
        return words
    
    def transcribe_text_only(
        self,
        audio_source: Union[str, Path]
    ) -> str:
        """
        Transcribe audio and return only the text.
        
        Args:
            audio_source: Path to audio file, URL, or Google Drive link
        
        Returns:
            Transcribed text
        """
        result = self.transcribe(audio_source, return_word_timestamps=False)
        return result['text']
    
    def _prepare_audio_source(self, audio_source: Union[str, Path]) -> str:
        """
        Prepare audio source (download if URL).
        
        Args:
            audio_source: Path to file, URL, or Google Drive link
        
        Returns:
            Path to local audio file
        """
        audio_source = str(audio_source)
        
        # Check if it's a URL
        if audio_source.startswith('http://') or audio_source.startswith('https://'):
            if 'drive.google.com' in audio_source:
                print("Downloading from Google Drive...")
                return download_audio_from_drive(audio_source)
            else:
                print("Downloading from URL...")
                return download_audio_from_url(audio_source)
        
        # It's a local file path
        return audio_source
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of language codes
        """
        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except:
            return ['en', 'ru', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pt', 'zh']
    
    def change_model(self, model_size: str):
        """
        Change to a different Whisper model size.
        
        Args:
            model_size: New model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Changing model from '{self.config.model_size}' to '{model_size}'...")
        
        # Unload current model
        if self.model:
            self.model.unload()
        
        # Update config and load new model
        self.config.model_size = model_size
        self._load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_size': self.config.model_size,
            'device': self.config.device,
            'language': self.config.language,
            'word_timestamps': self.config.word_timestamps,
        }
        
        if self.model:
            info.update(self.model.get_model_info())
        
        return info

