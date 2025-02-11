import librosa
import numpy as np
from config import SAMPLE_RATE, N_MFCC, N_MELS, N_FFT, HOP_LENGTH, DURATION

def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract combined features (MFCC and Mel spectrogram) from audio."""
    # Ensure audio is the correct length
    target_length = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Combine features
    features = np.concatenate([mfcc, mel_spec_db], axis=0)
    
    # Transpose to get time steps as the first dimension
    features = features.T
    
    # Expand dimensions for CNN input
    features = np.expand_dims(features, axis=-1)
    
    return features

def update_input_shape():
    """Update the INPUT_SHAPE in config based on the actual feature shape."""
    dummy_audio = np.zeros(int(SAMPLE_RATE * DURATION))
    dummy_features = extract_features(dummy_audio)
    
    import config
    config.INPUT_SHAPE = dummy_features.shape
    
    print(f"Updated INPUT_SHAPE: {config.INPUT_SHAPE}")

if __name__ == "__main__":
    update_input_shape()

