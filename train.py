import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
from glob import glob
from src.model import create_model
from src.data_collection import WakeWordDataGenerator
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_lists():
    hotword_files = glob(os.path.join(HOTWORD_DIR, '*.wav'))
    non_hotword_files = glob(os.path.join(NON_HOTWORD_DIR, '*.wav'))
    noise_files = glob(os.path.join(NOISE_DIR, '*.wav'))
    
    if not hotword_files:
        raise FileNotFoundError(f"No hotword files found in {HOTWORD_DIR}")
    if not non_hotword_files:
        raise FileNotFoundError(f"No non-hotword files found in {NON_HOTWORD_DIR}")
    
    return hotword_files, non_hotword_files, noise_files

def train_model():
    # Get file lists
    hotword_files, non_hotword_files, noise_files = get_file_lists()
    
    # Split data
    hotword_train, hotword_val = train_test_split(hotword_files, test_size=VALIDATION_SPLIT, random_state=42)
    non_hotword_train, non_hotword_val = train_test_split(non_hotword_files, test_size=VALIDATION_SPLIT, random_state=42)
    
    # Create data generators
    train_generator = WakeWordDataGenerator(hotword_train, non_hotword_train, noise_files)
    val_generator = WakeWordDataGenerator(hotword_val, non_hotword_val, noise_files)
    
    # Create and compile model
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Train the model
    logging.info("Starting training")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(MODELS_DIR, 'final_model.keras'))
    logging.info("Training completed successfully!")
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = train_model()
        print("Training completed successfully!")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")

