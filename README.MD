# Enhanced Hotword Detection System

## Overview

The Enhanced Hotword Detection System is an innovative, machine learning-based solution for recognizing specific trigger words or phrases in continuous audio streams. Developed by a team of ambitious first-semester B.Tech CSE students specializing in AI and ML, this project showcases the potential of emerging talent in the field of artificial intelligence and speech recognition.

This system provides high accuracy, low latency, and robust performance across various acoustic environments, making it suitable for a wide range of applications including voice assistants, automotive systems, industrial control, accessibility tools, smart home devices, security systems, educational technology, gaming, and telehealth.

## Key Features

- Advanced neural network architectures including CNN, RNN, CRNN, and Transformer for accurate hotword detection
- Robust audio preprocessing and feature extraction pipeline
- Real-time processing capabilities with low latency
- Adaptive noise cancellation and speaker normalization
- Advanced data augmentation techniques
- Support for continuous learning and model updates
- Customizable for multiple hotwords and languages
- Comprehensive API for easy integration into existing systems
- Extensive performance metrics and benchmarking tools
- Cross-platform compatibility (Windows, macOS, Linux)
- GPU optimization and mixed precision training
- Efficient data loading using TensorFlow's data pipeline
- Transfer learning capabilities using pre-trained models
- False positive reduction techniques

## System Architecture

The Enhanced Hotword Detection System consists of several key components:

1. **Audio Preprocessing Module**: Handles input audio streams, applying noise reduction, speaker normalization, and segmentation.
2. **Feature Extraction Engine**: Extracts relevant acoustic features including MFCCs and Mel spectrograms.
3. **Neural Network Model**: Multiple architecture options including CNN, RNN, CRNN, and Transformer, optimized for hotword detection.
4. **Post-processing Module**: Applies decision thresholding and smoothing to raw model outputs.
5. **Continuous Learning System**: Enables model updates with new data to improve performance over time.
6. **API Layer**: Provides interfaces for easy integration with other software systems.

## Development Team

This project is the result of collaborative efforts by a team of first-semester B.Tech CSE students specializing in AI and ML. The team structure includes:

- 1 Head Coder: Responsible for overall architecture and core algorithm development
- 5-6 Major Team Members: Focused on various aspects such as:
  - Data collection and preprocessing
  - Model training and optimization
  - Performance evaluation and benchmarking
  - API development and integration
  - Documentation and project management

As this is an ongoing project by students in their early stages of their academic journey, we acknowledge that the system may not be perfect and is continuously evolving. We welcome feedback, suggestions, and contributions from the community to help improve our project.

## Current Performance

While our system is still under development, we are continuously working to improve its performance. Current metrics:

- False Acceptance Rate (FAR): < 0.5%
- False Rejection Rate (FRR): < 3%
- Response Time: < 500ms

Please note that these metrics are subject to change as we refine our algorithms and expand our training data. Detailed benchmarking results and comparison with other systems are available in the `docs/benchmarks.md` file.

## Prerequisites

- Python 3.10.1
- TensorFlow 2.18.0
- CUDA-compatible GPU (recommended for training and high-performance inference)
- Additional libraries: librosa, numpy, scipy, soundfile, pyaudio, tqdm, scikit-learn, matplotlib

## Quick Start

1. Clone the repository:

2. Install the required dependencies: pip install -r requirements.txt

3. Prepare your dataset:
- Place hotword audio samples in `data/hotword/`
- Place non-hotword audio samples in `data/non_hotword/`

4. Run the main script: python main.py

## Project Structure

- `src/`
- `audio_utils.py`: Contains utility functions for audio processing and feature extraction
- `data_collection.py`: Handles data collection and preprocessing
- `features.py`: Handles feature extraction from audio
- `model.py`: Defines the base neural network architecture
- `train.py`: Main script for training the hotword detection model
- `evaluate.py`: Script for evaluating model performance
- `data/`
- `hotword/`: Directory for hotword audio samples
- `non_hotword/`: Directory for non-hotword audio samples
- `models/`: Directory for saving trained models and checkpoints
- `docs/`: Project documentation
- `config.py`: Configuration file for system parameters
- `main.py`: Main execution script

## Advanced Features

### GPU Optimization

The system is designed to utilize available GPUs for faster training and inference. It includes:
- Automatic GPU detection and configuration
- Mixed precision training for improved performance
- Memory growth settings for NVIDIA GPUs

### Data Augmentation

The system employs various data augmentation techniques to improve model robustness:
- Pitch shifting
- Time stretching
- Volume variation
- Room reverberation simulation
- Background noise injection

### Continuous Learning

The system supports incremental learning, allowing the model to adapt to new data over time:
- Checkpointing of model and optimizer states
- Resumable training sessions
- Daily training time limits to prevent overtraining

### Performance Optimization

- Efficient data loading using TensorFlow's data pipeline
- Multi-core CPU optimization
- Early stopping and learning rate reduction on plateau
- Class weight balancing for imbalanced datasets
- Quantization-aware training for latency optimization

### False Positive Reduction

Custom loss functions and training techniques are employed to minimize false activations, crucial for hotword detection systems.

## Customization

To train for a specific hotword (e.g., "kurma"):
1. Add multiple recordings of the hotword "kurma" to the `data/hotword/` directory.
2. Add various non-hotword audio samples to the `data/non_hotword/` directory.
3. Adjust the `MODEL_TYPE` in `config.py` if you want to experiment with different model architectures.
4. Run the `main.py` script to start the training process.

## Future Work

- Implement more sophisticated room simulation techniques for data augmentation
- Explore TTS-based data generation for expanding the dataset
- Implement additional advanced model architectures
- Further optimize the training process with techniques like curriculum learning
- Develop a user-friendly interface for model customization and deployment
- Implement a streaming inference pipeline for real-time detection

## License

[Insert your chosen license here]

## Acknowledgments

- TensorFlow team for their excellent deep learning framework
- The open-source community for various audio processing libraries

We welcome contributions and suggestions to help improve this project!