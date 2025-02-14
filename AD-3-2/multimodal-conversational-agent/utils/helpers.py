import os
import speech_recognition as sr
from gtts import gTTS
import librosa

def load_model(model_path):
    import torch
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_image(image):
    import cv2
    # Resize, normalize, and augment the image as needed
    image = cv2.resize(image, (224, 224))  # Example size for ViTs
    image = image / 255.0  # Normalize to [0, 1]
    return image

def preprocess_audio(audio):
    # Load audio file and extract features
    audio, _ = librosa.load(audio, sr=16000)  # Load with 16kHz sample rate
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return mfcc

def preprocess_text(text):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    tokens = tokenizer(text, return_tensors='pt')
    return tokens

def save_output(output, output_path):
    with open(output_path, 'w') as f:
        f.write(output)

def convert_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    
def text_to_speech(text, output_path='output.mp3'):
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path        