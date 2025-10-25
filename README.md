🐕 Dog Vision - Dog Breed Identification Project
================================================
A deep learning project that uses transfer learning with MobileNetV2 to identify dog breeds from images. This project achieves high accuracy on the Kaggle Dog Breed Identification dataset using TensorFlow and TensorFlow Hub.

📋 Project Summary
-----------------
Implements an end-to-end deep learning image classification system for 120 dog breeds. Uses transfer learning with MobileNetV2 (pre-trained on ImageNet), achieving impressive accuracy on a challenging multi-class task.

🏗️ Model Architecture
---------------------
- Base Model: MobileNetV2 (130_224) from TensorFlow Hub
- Model URL: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
- Input Shape: (None, 224, 224, 3) - RGB images, 224x224 px
- Output Shape: 120 classes (dog breeds)
- Layers: MobileNetV2 KerasLayer (non-trainable) + Dense Softmax output (trainable)
- Total Params: 5,552,953 (Trainable: 120,240; Non-trainable: 5,432,713)

📊 Dataset
---------
- Source: Kaggle Dog Breed Identification
- Training Images: 10,222
- Dog Breeds: 120
- Median Images/Breed: 82
- Test Images: 10,357
- Format: JPEG
- Split: Initial: 1,000 images; Full model: 10,222 images

🎯 Results & Performance
-----------------------
- 1,000 Image Model: Training Accuracy: 100%, Validation Accuracy: 66% (Best), Training stopped at Epoch 16.
- Full Model: Training Accuracy: 99.88% (Epoch 22), Loss: 0.0086, Early stopped at Epoch 22.
- Selected Milestones: 
    - Epoch 1: Acc 73.44%
    - Epoch 5: Acc 98.04%
    - Epoch 10: Acc 99.67%
    - Epoch 15: Acc 99.88%
    - Epoch 22: Acc 99.88%

🚀 Usage Steps
-------------
1. Setup Environment (Google Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. Install Dependencies:
    ```bash
    pip install tensorflow tensorflow-hub pandas numpy matplotlib scikit-learn
    ```
3. Prepare Data:
    - Images resize to 224x224, batch size 32, auto normalizes pixel values.
4. Create & Train Model:
    ```python
    model = create_model()
    # TensorBoard & EarlyStopping
    tensorboard = create_tensorboard_callback()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
    model.fit(x=train_data, epochs=NUM_EPOCHS, validation_data=val_data, validation_freq=1, callbacks=[tensorboard, early_stopping])
    ```
5. Make Predictions:
    ```python
    predictions = model.predict(val_data, verbose=1)
    pred_label = unique_breeds[np.argmax(predictions[0])]
    # For test dataset:
    test_predictions = model.predict(test_data, verbose=1)
    ```
6. Save & Load Model:
    ```python
    model.save('model_path.h5')
    loaded_model = tf.keras.models.load_model('model_path.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    ```

📦 Requirements
--------------
- tensorflow==2.x
- tensorflow-hub
- tf_keras
- pandas
- numpy
- scikit-learn
- matplotlib
- IPython
- google-colab (if using Colab)

📁 Project Structure
-------------------
```
dog-vision/
├── data/
│   ├── train/            # Training images
│   ├── test/             # Test images
│   ├── labels.csv        # Training labels
│   └── sample_submission.csv
├── logs/                 # TensorBoard logs
├── models/               # Saved models (.h5)
├── Dog_Vision_Full_Model.ipynb # Main notebook
└── README.md
```

🔑 Key Features
--------------
- Transfer learning (MobileNetV2)
- Automated image preprocessing & batching
- Visualization of predictions with confidence scores
- TensorBoard logging & early stopping
- Model persistence (save/load)
- Kaggle-compatible CSV outputs

📈 Training Details
------------------
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metric: Accuracy
- Batch Size: 32
- Image Processing: 0-1 normalization, Resize to 224x224, RGB
- Shuffle training, 20% validation split for experiments

📄 License
---------
MIT License – Free for educational and research use.

👤 Author
--------
Rageya Singh Raghuvanshi  
Email: rageyasinghr@gmail.com  
Colab notebook: Dog Vision Full Model  
Last updated: August 13, 2024

🙏 Acknowledgments
-----------------
- Kaggle Dog Breed Identification dataset
- TensorFlow Hub (MobileNetV2)
- Google Colab (GPU resources)

📝 Notes
-------
- Model can be improved via data augmentation
- Full dataset training: ~20–30 min with GPU
- Notebook includes extensive pred. visualization
- TensorBoard for training monitoring
