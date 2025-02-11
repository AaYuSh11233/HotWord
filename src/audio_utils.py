import numpy as np
import librosa
import scipy.signal as signal
from typing import List
import random
from config import SAMPLE_RATE, DURATION, TIME_STRETCH_RANGE, PITCH_SHIFT_RANGE, NOISE_FACTOR_RANGE

def load_and_preprocess_audio(file_path: str) -> np.ndarray:
    """Load and preprocess audio file."""
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
    else:
        audio = audio[:int(SAMPLE_RATE * DURATION)]
    return librosa.util.normalize(audio)

def apply_augmentation(audio: np.ndarray) -> List[np.ndarray]:
    """Apply various augmentations to the input audio."""
    augmenter = AudioAugmenter()
    augmented_audio = [
        audio,  # Original audio
        augmenter.time_stretch(audio),
        augmenter.pitch_shift(audio),
        augmenter.add_noise(audio),
        augmenter.add_room_reverb(audio)
    ]
    return augmented_audio

class AudioAugmenter:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
    
    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Apply time stretching."""
        rate = np.random.uniform(*TIME_STRETCH_RANGE)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting."""
        n_steps = np.random.uniform(*PITCH_SHIFT_RANGE)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random noise."""
        noise_factor = np.random.uniform(*NOISE_FACTOR_RANGE)
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def add_room_reverb(self, audio: np.ndarray, room_scale: float = 0.5) -> np.ndarray:
        """Simulate room reverberation."""
        room_ir = np.exp(-room_scale * np.linspace(0, 1, int(0.1 * self.sample_rate)))
        return signal.convolve(audio, room_ir, mode='same')

def mix_with_background(audio: np.ndarray, background: np.ndarray, snr_db: float = 10) -> np.ndarray:
    """Mix audio with background noise at a specified SNR."""
    audio_rms = np.sqrt(np.mean(audio**2))
    background_rms = np.sqrt(np.mean(background**2))
    snr_linear = 10**(snr_db/20)
    background_gain = audio_rms / (background_rms * snr_linear)
    mixed = audio + background_gain * background[:len(audio)]
    return librosa.util.normalize(mixed)

