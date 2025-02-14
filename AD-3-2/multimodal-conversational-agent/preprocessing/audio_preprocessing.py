from scipy.io import wavfile
import numpy as np
import librosa

def load_audio(file_path):
    """Load audio file and return the sample rate and audio data."""
    sample_rate, audio_data = wavfile.read(file_path)
    return sample_rate, audio_data

def preprocess_audio(audio_data, sample_rate):
    """Preprocess audio data: noise reduction, normalization, and MFCC extraction."""
    # Noise reduction (simple example using librosa)
    audio_data = librosa.effects.preemphasis(audio_data)

    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    
    return mfccs

def segment_audio(audio_data, segment_length):
    """Segment audio data into chunks of specified length."""
    segments = []
    for start in range(0, len(audio_data), segment_length):
        end = start + segment_length
        segments.append(audio_data[start:end])
    return segments

def extract_features(file_path):
    """Load, preprocess, and extract features from an audio file."""
    sample_rate, audio_data = load_audio(file_path)
    mfccs = preprocess_audio(audio_data, sample_rate)
    return mfccs
