from transformers import AutoTokenizer, AutoModel
import torch
import pyttsx3  # For text-to-speech (TTS) in Telugu

class TextModel:
    def __init__(self):
        # Load IndicBERT for Telugu text processing
        self.indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        self.indic_model = AutoModel.from_pretrained("ai4bharat/indic-bert")

        # Load XLM-RoBERTa for multilingual support (including Telugu)
        self.xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.xlm_model = AutoModel.from_pretrained("xlm-roberta-base")

    def process_text(self, text):
        # Process text using IndicBERT
        inputs = self.indic_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.indic_model(**inputs)
        return outputs.last_hidden_state

    def get_multilingual_embeddings(self, text):
        # Process text using XLM-RoBERTa
        inputs = self.xlm_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.xlm_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def generate_telugu_text(self, text):
        # Use XLM-RoBERTa embeddings to refine the Telugu output
        embeddings = self.get_multilingual_embeddings(text)
        # Placeholder: Process embeddings further if needed
        return text  # Return processed Telugu text (modify logic as needed)

def text_to_speech_telugu(text):
    """ Reads the output text in Telugu aloud using pyttsx3 """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('voice', 'te')  # Telugu language
    engine.say(text)
    engine.runAndWait()

def process_text(text):
    text_model = TextModel()
    processed_text = text_model.generate_telugu_text(text)

    print("Processed Telugu Output:", processed_text)
    text_to_speech_telugu(processed_text)  # Read output aloud in Telugu

    return processed_text

# Example usage
if __name__ == "__main__":
    sample_text = "తెలుగు భాష మాతృభాషగా వినిపిస్తుంది."
    process_text(sample_text)
