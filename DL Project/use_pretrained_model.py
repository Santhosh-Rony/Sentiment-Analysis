from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_sentiment_analyzer():
    """Initialize the sentiment analysis pipeline."""
    try:
        # Load pre-trained model (PyTorch backend - standard for Hugging Face models)
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
        logger.info("Successfully loaded the sentiment analysis model (PyTorch backend)")
        return sentiment_analyzer
    except Exception as e:
        logger.error(f"Error loading the model: {str(e)}")
        raise

def analyze_sentiment(text, analyzer):
    """Analyze the sentiment of given text."""
    try:
        result = analyzer(text)
        # Convert output to match your API format
        sentiment = "Positive" if result[0]["label"] == "POSITIVE" else "Negative"
        confidence = result[0]["score"]
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise

def analyze_batch(texts, analyzer):
    """Analyze sentiment for a batch of texts."""
    try:
        results = analyzer(texts)
        return [{
            "text": text,
            "sentiment": "Positive" if result["label"] == "POSITIVE" else "Negative",
            "confidence": result["score"]
        } for text, result in zip(texts, results)]
    except Exception as e:
        logger.error(f"Error analyzing batch: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the model
    analyzer = setup_sentiment_analyzer()
    test_texts = [
        "This movie was absolutely amazing!",
        "The service was terrible.",
        "It was okay, nothing special."
    ]
    
    print("\nTesting single text analysis:")
    result = analyze_sentiment(test_texts[0], analyzer)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    print("\nTesting batch analysis:")
    results = analyze_batch(test_texts, analyzer)
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")