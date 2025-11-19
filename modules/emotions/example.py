"""
Example usage of the Emotions classifier module.

This script demonstrates how to use the EmotionClassifier for
text emotion classification.
"""

from modules.emotions import EmotionClassifier, InferenceConfig


def format_result(text: str, result: dict):
    """Format prediction result for display."""
    print(f"\nТекст: \"{text}\"")
    print("=" * 70)
    print(f"Предсказанная эмоция: {result['emotion']}")
    print(f"Уверенность: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
    
    print("\nТоп-5 эмоций:")
    for i, pred in enumerate(result['top5'], 1):
        bar_length = int(pred['prob'] * 40)
        bar = "█" * bar_length
        print(f"  {i}. {pred['emotion']:15} {bar} {pred['prob']:.4f}")
    print("-" * 70)


def main():
    """Main example function."""
    
    print("=" * 70)
    print("Emotions Classifier - Example Usage")
    print("=" * 70)
    
    # Initialize classifier (loads model from HuggingFace)
    try:
        print("Инициализация классификатора эмоций...")
        print("(Модель будет загружена из HuggingFace при первом запуске)")
        classifier = EmotionClassifier()
        print("✅ Classifier initialized successfully\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        print("Убедитесь, что установлены необходимые библиотеки:")
        print("  pip install transformers torch")
        return
    
    # Example texts
    texts = [
        "Я так счастлив сегодня!",
        "Это ужасно, я в ярости!",
        "Мне очень грустно...",
        "Что за прекрасная погода!",
        "Ненавижу это место!",
        "Я немного волнуюсь перед экзаменом",
        "Спасибо тебе большое за помощь!",
        "Какой сюрприз!",
        "Я горжусь своими достижениями",
        "Мне любопытно, что будет дальше",
    ]
    
    # Classify each text
    for text in texts:
        result = classifier.predict(text)
        format_result(text, result)
    
    # Show all available emotions
    print("\n" + "=" * 70)
    print("Доступные эмоции (28 классов):")
    print("=" * 70)
    
    emotions = classifier.get_all_emotions()
    for i, emotion in enumerate(emotions, 1):
        print(f"{i:2d}. {emotion}")
    
    # Batch prediction example
    print("\n" + "=" * 70)
    print("Пакетная обработка:")
    print("=" * 70)
    
    batch_texts = [
        "Я люблю этот город",
        "Совершенно запутался",
        "Надеюсь, всё будет хорошо",
    ]
    
    results = classifier.predict_batch(batch_texts)
    
    print(f"\n{'Текст':<40} | {'Эмоция':<15} | {'Уверенность'}")
    print("-" * 70)
    for text, result in zip(batch_texts, results):
        text_short = text[:37] + "..." if len(text) > 40 else text
        print(f"{text_short:<40} | {result['emotion']:<15} | {result['confidence']:.4f}")


if __name__ == '__main__':
    main()

