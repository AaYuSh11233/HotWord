import os

# Audio processing
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160

# Model
INPUT_SHAPE = (None, N_MFCC + N_MELS, 1)  # Will be updated dynamically
NUM_CLASSES = 2
MODEL_TYPE = 'advanced'

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# Data
HOTWORD_DIR = os.path.join('data', 'kurma')
NON_HOTWORD_DIR = os.path.join('data', 'not_kurma')
NOISE_DIR = os.path.join('data', 'noise')
MODELS_DIR = 'models'

# Data Augmentation
AUGMENTATION_FACTOR = 3
TIME_STRETCH_RANGE = (0.8, 1.2)
PITCH_SHIFT_RANGE = (-2, 2)
NOISE_FACTOR_RANGE = (0.001, 0.015)

# Regularization
DROPOUT_RATE = 0.5
L2_LAMBDA = 0.001

# Model Evaluation
CONFIDENCE_THRESHOLD = 0.7

