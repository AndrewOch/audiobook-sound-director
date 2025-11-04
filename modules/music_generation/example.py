"""Example usage of the Music Generation module."""

from modules.music_generation import MusicGenerator, GeneratorConfig


def example_basic_usage():
    """Basic example of generating music from emotions."""
    print("=" * 60)
    print("Example 1: Basic Music Generation from Emotions")
    print("=" * 60)
    
    # Initialize generator with default config
    generator = MusicGenerator()
    
    # Define emotions (top-5 from emotion classifier)
    emotions = [
        ("joy", 0.85),
        ("contentment", 0.70),
        ("curiosity", 0.60),
        ("appreciation", 0.90),
        ("nostalgia", 0.50)
    ]
    
    # Generate 22 seconds of music
    audio = generator.generate_from_emotions(
        emotions=emotions,
        duration_seconds=22
    )
    
    # Save to file
    generator.save_audio(audio, "output/music_example1.wav")
    print(f"Generated audio shape: {audio.shape}")
    print(f"Sampling rate: {generator.get_sampling_rate()} Hz")
    print()


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 60)
    print("Example 2: Music Generation with Custom Config")
    print("=" * 60)
    
    # Create custom config
    config = GeneratorConfig(
        model_name="facebook/musicgen-small",
        device="auto",
        do_sample=True,
        guidance_scale=4.0,  # Higher guidance for more adherence to prompt
        tokens_per_second=50
    )
    
    # Initialize generator with custom config
    generator = MusicGenerator(config)
    
    # Define emotions (dramatic scene)
    emotions = [
        ("shock", 0.80),
        ("disapproval", 0.70),
        ("anger", 0.60),
        ("concern", 0.50),
        ("surprise", 0.40)
    ]
    
    # Generate 30 seconds of music
    audio = generator.generate_from_emotions(
        emotions=emotions,
        duration_seconds=30
    )
    
    # Save to file
    generator.save_audio(audio, "output/music_example2.wav")
    print(f"Generated audio shape: {audio.shape}")
    print()


def example_custom_prompt():
    """Example with custom prompt."""
    print("=" * 60)
    print("Example 3: Music Generation with Custom Prompt")
    print("=" * 60)
    
    generator = MusicGenerator()
    
    # Use custom prompt directly
    prompt = "Epic orchestral music with tension and drama"
    audio = generator.generate_from_prompt(
        prompt=prompt,
        duration_seconds=25
    )
    
    # Save to file
    generator.save_audio(audio, "output/music_example3.wav")
    print(f"Generated audio shape: {audio.shape}")
    print()


def example_with_emotion_classifier():
    """Example of integration with emotion classifier."""
    print("=" * 60)
    print("Example 4: Integration with Emotion Classifier")
    print("=" * 60)
    
    # Simulate emotion classifier output
    # In real usage, you would get this from EmotionClassifier
    text = "The sunset was absolutely beautiful, filling me with joy and peace."
    
    # Simulated emotion predictions
    emotion_predictions = [
        ("joy", 0.92),
        ("admiration", 0.75),
        ("optimism", 0.68),
        ("love", 0.54),
        ("gratitude", 0.48)
    ]
    
    print(f"Text: {text}")
    print(f"Predicted emotions: {emotion_predictions}")
    
    # Generate music
    generator = MusicGenerator()
    audio = generator.generate_from_emotions(
        emotions=emotion_predictions,
        duration_seconds=20
    )
    
    # Save to file
    generator.save_audio(audio, "output/music_example4.wav")
    print(f"Music generated based on emotional context!")
    print()


if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run examples
    print("\nMusic Generation Module - Examples\n")
    
    # Note: These examples require the model to be downloaded
    # and may take some time to run on CPU
    
    try:
        # Run basic example
        example_basic_usage()
        
        # Uncomment to run other examples:
        # example_custom_config()
        # example_custom_prompt()
        # example_with_emotion_classifier()
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("\nNote: Make sure you have:")
        print("1. Installed required packages: pip install transformers scipy torch")
        print("2. Sufficient disk space for model download (~1.5GB)")
        print("3. Internet connection for first-time model download")

