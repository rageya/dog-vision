# Dog-Vision

Dog-Vision is a deep learning project for dog breed classification using convolutional neural networks (CNN). It identifies and recognizes different dog breeds from images, providing accurate results for research and real-world applications.

## Features
- Classifies multiple dog breeds from input images
- Uses CNN architecture for image recognition
- Supports batch prediction and inference
- High accuracy results on benchmark datasets

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/rageya/dog-vision.git
   cd dog-vision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Make predictions:
   ```bash
   python inference.py --image path_to_your_image.jpg
   ```

## Requirements
- Python 3.x
- PyTorch
- numpy, matplotlib
- tqdm

## Results
- Accuracy: 91.8% on the Stanford Dogs dataset
- Model: EfficientNet-B0-based CNN

## License
MIT

## Author
rageya
