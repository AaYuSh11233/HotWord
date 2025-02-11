import numpy as np
import tensorflow as tf
from typing import List, Tuple
from src.audio_utils import load_and_preprocess_audio, apply_augmentation, mix_with_background
from src.features import extract_features
from config import BATCH_SIZE, AUGMENTATION_FACTOR

class WakeWordDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, hotword_files: List[str], non_hotword_files: List[str], noise_files: List[str], batch_size: int = BATCH_SIZE):
        self.hotword_files = hotword_files
        self.non_hotword_files = non_hotword_files
        self.noise_files = noise_files
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.hotword_files) * (1 + AUGMENTATION_FACTOR) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        for _ in range(self.batch_size):
            if np.random.rand() < 0.5:  # Hotword
                audio_file = np.random.choice(self.hotword_files)
                label = 1
            else:  # Non-hotword
                audio_file = np.random.choice(self.non_hotword_files)
                label = 0

            audio = load_and_preprocess_audio(audio_file)
            
            # Apply augmentation
            if np.random.rand() < AUGMENTATION_FACTOR / (1 + AUGMENTATION_FACTOR):
                augmented_audio = np.random.choice(apply_augmentation(audio))
            else:
                augmented_audio = audio

            # Mix with background noise
            if self.noise_files:
                noise_file = np.random.choice(self.noise_files)
                noise = load_and_preprocess_audio(noise_file)
                augmented_audio = mix_with_background(augmented_audio, noise, snr_db=np.random.uniform(5, 15))

            features = extract_features(augmented_audio)
            batch_x.append(features)
            batch_y.append(label)

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.hotword_files)
        np.random.shuffle(self.non_hotword_files)

