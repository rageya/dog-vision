# 🐕 Dog Vision - Dog Breed Identification Project

A deep learning project that uses transfer learning with MobileNetV2 to identify dog breeds from images. This project achieves high accuracy on the Kaggle Dog Breed Identification dataset using TensorFlow and TensorFlow Hub.

## 📋 Project Summary

This project implements an end-to-end deep learning image classification system to identify 120 different dog breeds from images. The model uses transfer learning with MobileNet V2 (130_224) pre-trained on ImageNet, achieving impressive accuracy on a challenging multi-class classification task.

## 🏗️ Model Architecture

- **Base Model**: MobileNetV2 (130_224) from TensorFlow Hub
- **Model URL**: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
- **Input Shape**: (None, 224, 224, 3) - RGB images resized to 224x224 pixels
- **Output Shape**: 120 classes (dog breeds)
- **Architecture Layers**:
  - Layer 1 (Input): KerasLayer (MobileNetV2 pre-trained) - 5,432,713 parameters (non-trainable)
  - Layer 2 (Output): Dense layer with softmax activation - 120,240 parameters (trainable)
- **Total Parameters**: 5,552,953 (21.18 MB)
  - Trainable: 120,240 (469.69 KB)
  - Non-trainable: 5,432,713 (20.72 MB)

## 📊 Dataset

- **Dataset**: Kaggle Dog Breed Identification Challenge
- **Training Images**: 10,222 images
- **Number of Breeds**: 120 unique dog breeds
- **Median Images per Breed**: 82 images
- **Test Images**: 10,357 images
- **Image Format**: JPEG files
- **Data Split**:
  - Initial experiments: 1,000 images (800 training, 200 validation)
  - Full model: 10,222 training images

## 🎯 Results & Performance

### Model on 1,000 Images:
- **Training Accuracy**: 100% (after 16 epochs)
- **Validation Accuracy**: 66% (best performance)
- **Validation Loss**: 1.2475
- **Training stopped at Epoch 16** due to early stopping (no improvement for 3 epochs)

### Full Model (All 10,222 Images):
- **Training Accuracy**: 99.88% (after 22 epochs)
- **Final Training Loss**: 0.0086
- **Training stopped at Epoch 22** due to early stopping
- **Model showed excellent convergence** with consistent improvement through epochs

### Training Progress Highlights:
- Epoch 1: Loss: 0.9879, Accuracy: 73.44%
- Epoch 5: Loss: 0.1018, Accuracy: 98.04%
- Epoch 10: Loss: 0.0306, Accuracy: 99.67%
- Epoch 15: Loss: 0.0154, Accuracy: 99.88%
- Epoch 22: Loss: 0.0086, Accuracy: 99.88% (Final)

## 🚀 Usage Steps

### 1. Setup Environment
```python
# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Set TensorFlow version
%tensorflow_version 2.x

# Import required libraries
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
```

### 2. Prepare Data
```python
# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_IMAGES = 1000  # Start with 1000 for experiments
NUM_EPOCHS = 100

# Load and process images
# Images are automatically:
# - Read from file paths
# - Decoded as JPEG with 3 color channels
# - Converted to float32 (values normalized from 0-255 to 0-1)
# - Resized to 224x224 pixels
# - Batched into groups of 32
```

### 3. Create and Train Model
```python
# Create model
model = create_model()

# Set up callbacks
tensorboard = create_tensorboard_callback()
early_stopping = tf_keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3
)

# Train model
model.fit(
    x=train_data,
    epochs=NUM_EPOCHS,
    validation_data=val_data,
    validation_freq=1,
    callbacks=[tensorboard, early_stopping]
)
```

### 4. Make Predictions
```python
# Make predictions on validation data
predictions = model.predict(val_data, verbose=1)

# Get predicted breed
pred_label = unique_breeds[np.argmax(predictions[0])]

# Make predictions on test data
test_predictions = model.predict(test_data, verbose=1)
```

### 5. Save and Load Model
```python
# Save model
model.save('model_path.h5')

# Load model
loaded_model = tf_keras.models.load_model(
    'model_path.h5',
    custom_objects={"KerasLayer": hub.KerasLayer}
)
```

## 📦 Requirements

```python
# Core Dependencies
tensorflow==2.x
tensorflow-hub
tf_keras

# Data Processing
pandas
numpy
scikit-learn

# Visualization
matplotlib
IPython

# Environment
google-colab  # If using Google Colab
```

## 📁 Project Structure

```
dog-vision/
├── data/
│   ├── train/          # Training images (10,222 images)
│   ├── test/           # Test images (10,357 images)
│   ├── labels.csv      # Training labels
│   └── sample_submission.csv
├── logs/               # TensorBoard logs
├── models/             # Saved models
│   ├── *_1000-images-mobilenetv2-Adam.h5
│   └── *_all-images-Adam.h5
├── Dog_Vision_Full_Model.ipynb  # Main notebook
└── README.md
```

## 🔑 Key Features

- **Transfer Learning**: Utilizes pre-trained MobileNetV2 for feature extraction
- **Data Preprocessing**: Automated image preprocessing pipeline with batching
- **Visualization**: Comprehensive visualization of predictions with confidence scores
- **Callbacks**: TensorBoard logging and early stopping for optimal training
- **Model Persistence**: Save and load trained models for future use
- **Kaggle Submission**: Export predictions in Kaggle-compatible CSV format

## 📈 Model Training Details

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Image Preprocessing**:
  - Normalization: 0-1 range
  - Resize: 224x224 pixels
  - Color channels: RGB (3 channels)
- **Data Augmentation**: Shuffling for training data
- **Validation Split**: 20% (800 train, 200 validation for 1000 images)

## 📄 License

MIT License - Feel free to use this project for educational and research purposes.

## 👤 Author

**Rageya Singh Raghuvanshi**
- Email: rageyasinghr@gmail.com
- Google Colab: [Dog Vision Full Model](https://colab.research.google.com/drive/1DW30AYcKDQRbNvVSR4fSlI1EFxbVvvZQ)
- Last Updated: August 13, 2024

## 🙏 Acknowledgments

- **Kaggle** for providing the Dog Breed Identification dataset
- **TensorFlow Hub** for pre-trained MobileNetV2 model
- **Google Colab** for providing free GPU resources

## 📝 Notes

- The model can be further improved with data augmentation techniques
- Training on full dataset takes approximately 20-30 minutes with GPU acceleration
- The notebook includes comprehensive visualization tools for model predictions
- TensorBoard integration allows for detailed training monitoring
