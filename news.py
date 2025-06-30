from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the FinBERT model and tokenizer
model_name = "./model"  # This appears to be the model from the files you shared
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    """
    Analyze the sentiment of financial news text
    Returns: 
        dict: {'sentiment': 'Positive/Negative/Neutral', 'confidence': float}
    """
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class and confidence
    confidence, predicted_class = torch.max(probs, dim=-1)
    
    # Map class index to label
    class_names = ['Positive', 'Negative', 'Neutral']
    sentiment = class_names[predicted_class.item()]
    
    return {
        'sentiment': sentiment,
        'confidence': confidence.item()
    }

# Example usage
if __name__ == "__main__":
    news_articles = [
        "chinese financial technology firm ant group said on monday that it has increased investment in research and development fro 4 consecutive years,since 2021, reaching a record high of 23.45 billion yuan in 2024"
    ]
    
    for article in news_articles:
        result = analyze_sentiment(article)
        print(f"Article: {article}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print("-" * 80)