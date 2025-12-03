"""
LLM-based Emotion Classifier using Scaleway API

This module provides emotion classification using LLM models via Scaleway API.
It can be used as an alternative to the local RuBERT model for more accurate
emotion detection, especially for complex Russian texts.
"""

import time
import re
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class LLMEmotionConfig:
    """Configuration for LLM-based emotion classification."""
    
    api_key: str
    project_id: str
    model: str = "llama-3.3-70b-instruct"  # Default model
    base_url: Optional[str] = None  # Auto-constructed if None
    max_tokens: int = 50
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        """Construct base_url if not provided."""
        if self.base_url is None:
            self.base_url = f"https://api.scaleway.ai/{self.project_id}/v1"


# System prompt for emotion analysis
SYSTEM_PROMPT = """Perform a detailed emotional analysis of the Russian text provided below.

1.  Carefully read and analyze the given text.

2.  Your task is to identify the single most dominant emotion from the provided list.

3.  You must also assign a probability score (0-100%) that reflects your confidence in this emotion being the primary one.

List of Emotions:
amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral.

Write your final answer strictly in the json format: {"emotion": emotion, "score": score}, do not give any explanations or additional info, don't add a % sign"""


class LLMEmotionClassifier:
    """
    Emotion classifier using LLM models via Scaleway API.
    
    This classifier uses large language models (LLM) to analyze emotions in text,
    which can provide more nuanced understanding compared to local models.
    """
    
    def __init__(self, config: LLMEmotionConfig):
        """
        Initialize the LLM emotion classifier.
        
        Args:
            config: Configuration with API credentials and model settings
        """
        if OpenAI is None:
            raise ImportError(
                "openai library is required. "
                "Install it with: pip install openai"
            )
        
        self.config = config
        self.logger = logging.getLogger("audiobook.emotions.llm")
        
        # Initialize OpenAI client for Scaleway
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
        
        self.logger.info(f"LLM Emotion Classifier initialized with model: {config.model}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict emotion for the input text using LLM.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results:
            {
                'emotion': str,      # Top-1 emotion
                'confidence': float, # Top-1 probability (0-1)
                'top5': [            # Top-5 predictions (only top-1 for LLM)
                    {'emotion': str, 'prob': float},
                ]
            }
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                presence_penalty=self.config.presence_penalty
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = self._parse_response(content)
            
            self.logger.debug(f"LLM prediction completed in {latency:.2f}s: {result['emotion']} ({result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.logger.error(f"Error in LLM emotion prediction: {str(e)} (latency: {latency:.2f}s)")
            
            # Return neutral fallback
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'top5': [{'emotion': 'neutral', 'prob': 0.0}],
                'error': str(e)
            }
    
    def _parse_response(self, content: str) -> Dict:
        """
        Parse LLM response to extract emotion and score.
        
        Args:
            content: Raw response from LLM
            
        Returns:
            Parsed result dictionary
        """
        # Try to extract JSON from response
        # LLM might wrap JSON in markdown code blocks or add extra text
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content
        
        try:
            data = json.loads(json_str)
            emotion = data.get("emotion", "neutral").lower()
            score = data.get("score", 0)
            
            # Convert score from 0-100 to 0-1 if needed
            if score > 1.0:
                score = score / 100.0
            score = max(0.0, min(1.0, float(score)))
            
            return {
                'emotion': emotion,
                'confidence': score,
                'top5': [{'emotion': emotion, 'prob': score}],
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(f"Failed to parse LLM response: {content}. Error: {e}")
            # Try to extract emotion name from text as fallback
            emotion_match = re.search(r'"(?:emotion|emotion_name)":\s*"([^"]+)"', content, re.IGNORECASE)
            if emotion_match:
                emotion = emotion_match.group(1).lower()
            else:
                emotion = "neutral"
            
            return {
                'emotion': emotion,
                'confidence': 0.5,  # Default confidence
                'top5': [{'emotion': emotion, 'prob': 0.5}],
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def get_all_emotions(self) -> List[str]:
        """Get list of all emotion labels."""
        return [
            "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust",
            "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
            "love", "nervousness", "optimism", "pride", "realization", "relief",
            "remorse", "sadness", "surprise", "neutral"
        ]

