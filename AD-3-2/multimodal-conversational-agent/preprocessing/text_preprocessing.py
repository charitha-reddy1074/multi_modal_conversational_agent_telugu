from transformers import AutoTokenizer, AutoModel
import torch
import re

# Load the tokenizer and model for IndicBERT or XLM-RoBERTa
MODEL_NAME = "ai4bharat/indic-bert"  # Change to "xlm-roberta-base" if using XLM-RoBERTa
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def clean_text(text):
    """Clean the input text by removing unwanted characters and normalizing."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading and trailing spaces
    return text

def tokenize_text(text):
    """Tokenize the cleaned text and return input IDs and attention masks."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return inputs['input_ids'], inputs['attention_mask']

def embed_text(text):
    """Generate embeddings for the input text using the model."""
    cleaned_text = clean_text(text)
    input_ids, attention_mask = tokenize_text(cleaned_text)
    
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask=attention_mask).last_hidden_state
    return embeddings

# Example usage
if __name__ == "__main__":
    sample_text = "మీరు ఎలా ఉన్నారు?"  # "How are you?" in Telugu
    embeddings = embed_text(sample_text)
    print(embeddings)