"""
Example usage of the Speech Recognition module.

This script demonstrates how to use the SpeechRecognizer
for transcribing audio with word-level timestamps.
"""

from modules.speech_recognition import SpeechRecognizer, RecognizerConfig


def example_basic():
    """Basic transcription example."""
    print("=" * 70)
    print("Basic Transcription Example")
    print("=" * 70)
    
    # Initialize recognizer
    recognizer = SpeechRecognizer()
    
    # Note: You need to provide an actual audio file
    audio_file = "audio.mp3"
    
    print(f"\nℹ️  To run this example, please provide an audio file:")
    print(f"   recognizer.transcribe('{audio_file}')")
    print(f"\nExample with dummy data:\n")
    
    # Simulated result structure
    result = {
        'text': "Пример транскрипции аудио файла",
        'language': 'ru',
        'duration': 5.2,
        'segments': [
            {
                'text': 'Пример транскрипции',
                'start': 0.0,
                'end': 2.5,
                'words': [
                    {'text': 'Пример', 'start': 0.0, 'end': 0.8},
                    {'text': 'транскрипции', 'start': 0.9, 'end': 2.5},
                ]
            },
            {
                'text': 'аудио файла',
                'start': 2.6,
                'end': 5.2,
                'words': [
                    {'text': 'аудио', 'start': 2.6, 'end': 3.5},
                    {'text': 'файла', 'start': 3.6, 'end': 5.2},
                ]
            }
        ]
    }
    
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Duration: {result['duration']:.2f}s\n")
    
    print("Segments:")
    for i, segment in enumerate(result['segments'], 1):
        print(f"  {i}. [{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
    
    print("\nWord timestamps:")
    for segment in result['segments']:
        for word in segment['words']:
            print(f"  {word['text']}: {word['start']:.2f}s - {word['end']:.2f}s")


def example_different_models():
    """Example with different model sizes."""
    print("\n" + "=" * 70)
    print("Different Model Sizes Example")
    print("=" * 70)
    
    models = ['tiny', 'base', 'small', 'medium']
    
    for model_size in models:
        config = RecognizerConfig(model_size=model_size)
        print(f"\n{model_size.upper()} model:")
        print(f"  Parameters: {RecognizerConfig.__annotations__}")
        print(f"  Best for: ", end="")
        
        if model_size == 'tiny':
            print("быстрая обработка, низкое качество")
        elif model_size == 'base':
            print("баланс скорости и качества ⭐")
        elif model_size == 'small':
            print("хорошее качество, средняя скорость")
        elif model_size == 'medium':
            print("очень хорошее качество, медленная")


def example_config_options():
    """Example with different configuration options."""
    print("\n" + "=" * 70)
    print("Configuration Options Example")
    print("=" * 70)
    
    configs = [
        {
            'name': 'Fast & Low Quality',
            'config': RecognizerConfig(model_size='tiny', word_timestamps=False)
        },
        {
            'name': 'Balanced (Default)',
            'config': RecognizerConfig(model_size='base', word_timestamps=True)
        },
        {
            'name': 'High Quality',
            'config': RecognizerConfig(model_size='medium', beam_size=10)
        },
        {
            'name': 'Translation to English',
            'config': RecognizerConfig(task='translate', language='ru')
        },
    ]
    
    for cfg_info in configs:
        print(f"\n{cfg_info['name']}:")
        cfg = cfg_info['config']
        print(f"  Model: {cfg.model_size}")
        print(f"  Task: {cfg.task}")
        print(f"  Word timestamps: {cfg.word_timestamps}")
        if cfg.task == 'translate':
            print(f"  (Translates {cfg.language} → English)")


def example_download_sources():
    """Example with different audio sources."""
    print("\n" + "=" * 70)
    print("Different Audio Sources Example")
    print("=" * 70)
    
    sources = [
        {
            'type': 'Local file',
            'source': 'audio.mp3',
            'description': 'Audio file on your computer'
        },
        {
            'type': 'URL',
            'source': 'https://example.com/audio.mp3',
            'description': 'Direct download link'
        },
        {
            'type': 'Google Drive',
            'source': 'https://drive.google.com/file/d/FILE_ID/view',
            'description': 'Shared Google Drive file'
        },
    ]
    
    print("\nSupported audio sources:")
    for source_info in sources:
        print(f"\n{source_info['type']}:")
        print(f"  Source: {source_info['source']}")
        print(f"  Description: {source_info['description']}")
        print(f"  Usage: recognizer.transcribe('{source_info['source']}')")


def example_save_formats():
    """Example of saving transcription in different formats."""
    print("\n" + "=" * 70)
    print("Save Transcription Formats Example")
    print("=" * 70)
    
    formats = [
        ('txt', 'Plain text file'),
        ('json', 'JSON with all metadata'),
        ('srt', 'SubRip subtitle format'),
        ('vtt', 'WebVTT subtitle format'),
    ]
    
    print("\nAvailable save formats:")
    for fmt, description in formats:
        print(f"  {fmt.upper()}: {description}")
    
    print("\nExample usage:")
    print("  from modules.speech_recognition import save_transcription_to_file")
    print("  ")
    print("  result = recognizer.transcribe('audio.mp3')")
    print("  save_transcription_to_file(result, 'output.txt', format='txt')")


def main():
    """Main example function."""
    print("\n" + "=" * 70)
    print("Speech Recognition Module - Examples")
    print("=" * 70)
    
    # Run examples
    example_basic()
    example_different_models()
    example_config_options()
    example_download_sources()
    example_save_formats()
    
    # Model information
    print("\n" + "=" * 70)
    print("Model Information")
    print("=" * 70)
    
    from modules.speech_recognition import MODEL_INFO
    
    print("\nAvailable Whisper Models:\n")
    for model_name, info in MODEL_INFO.items():
        print(f"{model_name.upper()}:")
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
        print()
    
    # Supported languages
    print("=" * 70)
    print("Supported Languages")
    print("=" * 70)
    
    print("\nWhisper supports 99 languages including:")
    languages = [
        ('ru', 'Russian'),
        ('en', 'English'),
        ('de', 'German'),
        ('es', 'Spanish'),
        ('fr', 'French'),
        ('it', 'Italian'),
        ('ja', 'Japanese'),
        ('ko', 'Korean'),
        ('zh', 'Chinese'),
        ('pt', 'Portuguese'),
    ]
    
    for code, name in languages:
        print(f"  {code} - {name}")
    
    print("\n... and many more!")
    
    # Final notes
    print("\n" + "=" * 70)
    print("Getting Started")
    print("=" * 70)
    
    print("\n1. Install dependencies:")
    print("   pip install openai-whisper torch pydub")
    print("\n2. Install ffmpeg:")
    print("   brew install ffmpeg  # macOS")
    print("   sudo apt install ffmpeg  # Linux")
    print("\n3. Use the recognizer:")
    print("   from modules.speech_recognition import SpeechRecognizer")
    print("   recognizer = SpeechRecognizer()")
    print("   result = recognizer.transcribe('your_audio.mp3')")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

