#pip install -U openai-whisper
#pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer and model from local directory
model_path = r"D:\text-detox_model\xlmr-large-toxicity-classifier-v2"
  # replace with your actual folder path if different

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Sample text inputs
texts = [
    "I hate you",  # Inappropriate
    "You are amazing!",  # Appropriate
    "This is disgusting",  # Inappropriate
]

# Tokenize and get predictions
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        toxicity_score = probs[0][1].item()

    print(f"Text: {text}")
    print(f"Toxicity Score: {toxicity_score:.4f} ({'Toxic' if toxicity_score > 0.5 else 'Non-Toxic'})\n")
