import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_collection import WakeWordDataGenerator
from config import HOTWORD_DIR, MODELS_DIR, BATCH_SIZE, CONFIDENCE_THRESHOLD, NOISE_DIR, NON_HOTWORD_DIR

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def evaluate_model(model, test_generator):
    # Predict on test data
    y_pred = model.predict(test_generator)
    y_true = np.concatenate([y for x, y in test_generator])
    
    # Convert probabilities to class predictions
    y_pred_classes = (y_pred[:, 1] > CONFIDENCE_THRESHOLD).astype(int)
    
    # Print classification report
    print(classification_report(y_true, y_pred_classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def main():
    # Load test data
    hotword_test_files = [f for f in os.listdir(HOTWORD_DIR) if f.startswith('test_')]
    non_hotword_test_files = [f for f in os.listdir(NON_HOTWORD_DIR) if f.startswith('test_')]
    noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]
    
    test_generator = WakeWordDataGenerator(hotword_test_files, non_hotword_test_files, noise_files, batch_size=BATCH_SIZE)
    
    # Load the best model
    model = load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
    
    # Evaluate the model
    evaluate_model(model, test_generator)

if __name__ == "__main__":
    main()

