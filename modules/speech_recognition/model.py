"""
Whisper Model Wrapper

This module provides a wrapper around OpenAI's Whisper model
for speech recognition.
"""

from typing import Optional, Dict, Any


class WhisperModel:
    """
    Wrapper around Whisper model for speech recognition.
    
    This class handles loading and managing the Whisper model,
    providing a clean interface for transcription.
    """
    
    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'cpu',
        download_root: Optional[str] = None
    ):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Size of the model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cpu', 'cuda', 'mps')
            download_root: Custom directory for model cache (optional)
        """
        self.model_size = model_size
        self.device = device
        self.download_root = download_root
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for speech recognition. "
                "Install it with: pip install openai-whisper"
            )
        
        print(f"Loading Whisper model '{self.model_size}' on '{self.device}'...")
        
        try:
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=self.download_root
            )
        except Exception as e:
            # Some PyTorch versions have incomplete MPS ops; fallback to CPU
            if self.device == 'mps':
                print(f"MPS load failed, falling back to CPU: {e}")
                self.device = 'cpu'
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=self.download_root
                )
            else:
                raise
        
        print(f"✓ Model loaded successfully")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = 'transcribe',
        word_timestamps: bool = True,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'ru', 'en'). None for auto-detection.
            task: 'transcribe' or 'translate' (translate to English)
            word_timestamps: Extract word-level timestamps
            temperature: Sampling temperature (0 = greedy, higher = more random)
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling
            patience: Beam search patience factor
            verbose: Print progress during transcription
            **kwargs: Additional arguments for whisper.transcribe()
        
        Returns:
            Dictionary with transcription results:
            {
                'text': str,  # Full transcription
                'segments': list,  # Segments with timestamps
                'language': str,  # Detected/specified language
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare transcription options
        options = {
            'language': language,
            'task': task,
            'word_timestamps': word_timestamps,
            'temperature': temperature,
            'beam_size': beam_size,
            'best_of': best_of,
            'patience': patience,
            'verbose': verbose,
            **kwargs
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        # Transcribe with safe fallback from MPS to CPU if needed
        try:
            result = self.model.transcribe(audio_path, **options)
            return result
        except NotImplementedError as e:
            if self.device == 'mps':
                print(f"MPS transcribe failed, retrying on CPU: {e}")
                self.unload()
                self.device = 'cpu'
                self._load_model()
                result = self.model.transcribe(audio_path, **options)
                return result
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'download_root': self.download_root,
            'is_loaded': self.model is not None,
        }
    
    def unload(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Clear CUDA cache if on GPU
            if self.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
            
            print("✓ Model unloaded")


