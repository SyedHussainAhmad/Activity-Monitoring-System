# Human Activity Recognition: Open Door vs Rub Hands Classification

A machine learning project for classifying human activities using accelerometer and gyroscope sensor data. This binary classification system distinguishes between "Open Door" and "Rub Hands" activities using traditional machine learning and deep learning approaches.

## Project Overview

This project implements a human activity recognition (HAR) system that processes motion sensor data to classify two specific activities:
- **Open Door** (Class 0)
- **Rub Hands** (Class 1)

The system extracts statistical features from raw sensor data and employs both traditional machine learning (Logistic Regression) and deep learning (Neural Network) approaches for classification.

## 📊 Dataset

- **Training Data**: 87 samples
- **Testing Data**: 90 samples
- **Sensors**: 2 (Accelerometer + Gyroscope)
- **Time Series Length**: 268 time steps per sample
- **Axes**: 3 (X, Y, Z) per sensor

### Data Structure
```
├── train_MSAccelerometer_OpenDoor_RubHands.npy  # Training accelerometer data
├── train_MSGyroscope_OpenDoor_RubHands.npy     # Training gyroscope data
├── train_labels_OpenDoor_RubHands.npy          # Training labels
├── test_MSAccelerometer_OpenDoor_RubHands.npy  # Testing accelerometer data
├── test_MSGyroscope_OpenDoor_RubHands.npy      # Testing gyroscope data
└── test_labels_OpenDoor_RubHands.npy           # Testing labels
```

## 🔧 Features Engineering

The system extracts 16 statistical features per sensor (32 total features × 3 axes = 48 final features):

### Per Sensor Features:
1. **Mean** - Average signal value
2. **Maximum** - Peak signal value
3. **Minimum** - Lowest signal value
4. **Standard Deviation** - Signal variability
5. **Zero Crossings** - Number of sign changes
6. **20th Percentile** - Lower quartile boundary
7. **50th Percentile** - Median value
8. **80th Percentile** - Upper quartile boundary

## Model Architecture

### 1. Neural Network Model
```python
Architecture:
Input Layer: 48 features
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 64 neurons (ReLU)
Hidden Layer 3: 32 neurons (ReLU)
Output Layer: 1 neuron (Sigmoid)
```

**Hyperparameters:**
- Learning Rate: 0.01
- Epochs: 150
- Batch Size: 50
- Optimizer: SGD with momentum (0.9)
- Loss Function: Binary Cross Entropy

### 2. Logistic Regression
- Solver: liblinear
- Traditional statistical approach for comparison

## 📈 Results

| Model | Accuracy | Weighted F1-Score | Average F1-Score |
|-------|----------|------------------|------------------|
| **Neural Network** | **97.78%** | - | - |
| **Logistic Regression** | 93.33% | 0.9334 | 0.9333 |

### Confusion Matrix (Logistic Regression)

|                         | **Predicted: Open Door** | **Predicted: Rub Hands** |
|:-----------------------:|:------------------------:|:------------------------:|
| **Actual: Open Door**   | 43                       | 5                        |
| **Actual: Rub Hands**   | 1                        | 41                       |


True Positives (TP) = 43

True Negatives (TN) = 41

False Positives (FP) = 1

False Negatives (FN) = 5


## Getting Started

### Prerequisites
```bash
pip install numpy scikit-learn torch matplotlib
```

### Installation
```bash
git clone https://github.com/SyedHussainAhmad/human-activity-recognition
cd human-activity-recognition
```

### Usage
```python
# Load and preprocess data
python data_preprocessing.py

# Train models
python train_models.py

# Evaluate performance
python evaluate_models.py
```

## 📁 Project Structure
```
├── data/                          # Dataset files
├── models/                        # Trained model files
├── src/
│   ├── data_preprocessing.py      # Data loading and feature extraction
│   ├── neural_network.py         # Deep learning model
│   ├── logistic_regression.py    # Traditional ML model
│   └── evaluation.py             # Model evaluation utilities
├── notebooks/                     # Jupyter notebooks
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## Technical Details

### Feature Extraction Process
1. Load raw accelerometer and gyroscope time series data
2. Calculate 8 statistical features per sensor per axis
3. Reshape feature matrix from (samples, features, axes) to (samples, features×axes)
4. Normalize features for optimal model performance

### Model Training
- **Neural Network**: Gradient descent optimization with momentum
- **Logistic Regression**: Maximum likelihood estimation
- **Evaluation**: Accuracy, F1-scores (weighted, macro, per-class), confusion matrix

## Future Improvements

- [ ] Implement additional feature extraction techniques (frequency domain, wavelet transforms)
- [ ] Experiment with LSTM/GRU networks for temporal modeling
- [ ] Add cross-validation for more robust evaluation
- [ ] Extend to multi-class activity recognition
- [ ] Implement real-time inference capabilities
- [ ] Add data augmentation techniques

## Performance Analysis

The neural network achieves superior performance (97.78% accuracy) compared to logistic regression (93.33% accuracy), demonstrating the effectiveness of deep learning for this HAR task. The high F1-scores indicate balanced performance across both activity classes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 👤 Author

**Syed Hussain Ahmad**
- GitHub: [@SyedHussainAhmad](https://github.com/SyedHussainAhmad)
- LinkedIn: [https://www.linkedin.com/in/syedhussainahmad/]

## Acknowledgments

- Thanks to the contributors of the sensor data collection
- Inspired by advances in human activity recognition research
- Built with PyTorch and scikit-learn

---

⭐ **Star this repository if you found it helpful!**
