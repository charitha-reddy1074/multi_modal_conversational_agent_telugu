import torchaudio
import torch
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from gtts import gTTS
import soundfile as sf

class AudioModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("anuragshas/wav2vec2-large-xlsr-53-telugu")
        self.model = Wav2Vec2ForCTC.from_pretrained("anuragshas/wav2vec2-large-xlsr-53-telugu")

    def transcribe(self, audio_input):
        # Process audio input
        input_values = self.processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the ids to Telugu text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

def process_audio(audio_path):
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)

        # Flatten the waveform
        audio_input = waveform.squeeze().numpy()

        # Get transcription
        audio_model = AudioModel()
        transcription = audio_model.transcribe(audio_input)

        return transcription
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def text_to_speech_telugu(text, output_path="output_telugu.wav"):
    tts = gTTS(text, lang='te')  # Generate Telugu speech
    tts.save(output_path)
    return output_path

def audio_to_text_and_speech(audio_path):
    transcription = process_audio(audio_path)
    if transcription is None:
        return None, None

    # Generate Telugu speech output
    output_audio_path = text_to_speech_telugu(transcription)

    return transcription, output_audio_path

# Example usage
if __name__ == "__main__":
    audio_path = "example.wav"  # Replace with your actual audio file
    transcription, speech_output = audio_to_text_and_speech(audio_path)
    if transcription and speech_output:
        print("Transcription:", transcription)
        print("Generated Speech File:", speech_output)
    else:
        print("Failed to process the audio file.")