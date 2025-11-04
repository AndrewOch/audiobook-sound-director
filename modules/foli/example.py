"""
Example usage of the Foli classifier module.

This script demonstrates how to use both PyTorch and ONNX versions
of the classifier.
"""

from modules.foli import FoliClassifierPyTorch, FoliClassifierONNX, InferenceConfig


def format_result(text: str, result: dict):
    """Format prediction result for display."""
    print(f"\nТекст: {text}")
    print("=" * 60)
    
    for ch in ['ch1', 'ch2', 'ch3']:
        print(f"\n{ch.upper()}:")
        print(f"  Топ-1: {result[ch]['class']} (вероятность: {result[ch]['prob']:.3f})")
        print(f"  Топ-5:")
        for i, pred in enumerate(result[ch]['top5'], 1):
            print(f"    {i}. {pred['class']}: {pred['prob']:.3f}")


def main():
    """Main example function."""
    
    # Example texts
    texts = [
        "Слышишь, как в подъезде глухо хлопнула дверь?",
        "Записываю подкаст, а за окном раз в минуту протяжно гудит теплоход.",
        "В кухне кипит чайник, и в тишине еле звенит ложка о край кружки.",
        "На стадионе диктор объявил составы, толпа загудела.",
        "В деревне петух прокричал так резко, что кошка сорвалась с подоконника.",
    ]
    
    print("=" * 60)
    print("Foli Classifier - Example Usage")
    print("=" * 60)
    
    # PyTorch inference
    print("\n\n### Using PyTorch Backend ###\n")
    classifier_pt = FoliClassifierPyTorch()
    
    for text in texts[:2]:  # Show detailed results for first 2
        result = classifier_pt.predict(text)
        format_result(text, result)
    
    # ONNX inference (if available)
    print("\n\n### Using ONNX Backend ###\n")
    try:
        classifier_onnx = FoliClassifierONNX()
        
        for text in texts[2:4]:  # Show detailed results for next 2
            result = classifier_onnx.predict(text)
            format_result(text, result)
    except Exception as e:
        print(f"ONNX inference not available: {e}")
    
    # Comparison table
    print("\n\n### Comparison Table (All Texts) ###\n")
    print(f"{'Text':<50} | {'ch1':<20} | {'ch2':<20} | {'ch3':<20}")
    print("-" * 115)
    
    for text in texts:
        result = classifier_pt.predict(text)
        text_short = text[:47] + "..." if len(text) > 50 else text
        print(
            f"{text_short:<50} | "
            f"{result['ch1']['class']:<20} | "
            f"{result['ch2']['class']:<20} | "
            f"{result['ch3']['class']:<20}"
        )


if __name__ == '__main__':
    main()

