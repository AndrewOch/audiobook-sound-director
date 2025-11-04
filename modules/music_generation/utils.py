"""Utility functions for Music Generation module."""

from typing import List, Tuple


def format_emotions_prompt(
    emotions: List[Tuple[str, float]],
    duration_seconds: int = 22,
    base_prompt: str = "Generate {duration} second background track for an audiobook using this emotions:"
) -> str:
    """Format emotion predictions into a text prompt for music generation.
    
    Args:
        emotions: List of tuples (emotion_name, probability)
        duration_seconds: Duration of music to generate in seconds
        base_prompt: Base prompt template with {duration} placeholder
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> emotions = [("joy", 0.85), ("contentment", 0.70), ("curiosity", 0.60)]
        >>> prompt = format_emotions_prompt(emotions, duration_seconds=30)
        >>> print(prompt)
        Generate 30 second background track for an audiobook using this emotions: 1. Joy - 85% 2. Contentment - 70% 3. Curiosity - 60%
    """
    # Format the base prompt with duration
    prompt_text = base_prompt.format(duration=duration_seconds)
    
    # Format emotions list
    emotions_text = " ".join([
        f"{i+1}. {name.capitalize()} - {int(prob * 100)}%"
        for i, (name, prob) in enumerate(emotions)
    ])
    
    return f"{prompt_text} {emotions_text}"


def calculate_max_tokens(duration_seconds: int, tokens_per_second: int = 50) -> int:
    """Calculate max_new_tokens for generation based on desired duration.
    
    Args:
        duration_seconds: Desired audio duration in seconds
        tokens_per_second: Number of tokens per second (default: 50 for MusicGen)
        
    Returns:
        Number of tokens to generate
        
    Example:
        >>> calculate_max_tokens(22)
        1100
    """
    return duration_seconds * tokens_per_second

