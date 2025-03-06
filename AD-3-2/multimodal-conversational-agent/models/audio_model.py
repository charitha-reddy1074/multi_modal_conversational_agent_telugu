import torchaudio
import torch
import soundfile as sf
import os
import pyaudio
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from gtts import gTTS
import simpleaudio as sa

class AudioModel:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-te")  # Telugu Model

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

def record_audio(filename="live_audio.wav", duration=5, sample_rate=16000):
    """Records audio from the microphone."""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)
    
    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

def process_audio(audio_path):
    """Loads an audio file and transcribes it to Telugu text."""
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

def text_to_speech_telugu(text, output_path="output_telugu.wav"):
    """Converts text to speech in Telugu and plays the audio."""
    tts = gTTS(text, lang='te')
    tts.save(output_path)

    # Play the generated audio
    wave_obj = sa.WaveObject.from_wave_file(output_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    return output_path

def audio_to_text_and_speech():
    """Records, transcribes, converts to speech, and plays the output audio."""
    recorded_audio = record_audio()
    transcription = process_audio(recorded_audio)
    print("Transcription:", transcription)

    # Generate and play Telugu speech output
    output_audio_path = text_to_speech_telugu(transcription)
    print("Generated Speech File:", output_audio_path)

    return transcription, output_audio_path

# Example usage
if __name__ == "__main__":
    audio_to_text_and_speech()
