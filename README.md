# Hand Gesture Recognition using CNN and MediaPipe

This project focuses on real-time recognition of static hand gestures using computer vision and convolutional neural networks (CNN).
Hand landmarks are extracted from a camera stream using MediaPipe Hands, and gestures are classified based on their spatial configuration.

The project covers the complete pipeline from data collection to model training, evaluation, and real-time inference.

---

## Features

- Real-time hand landmark detection using MediaPipe Hands
- Static hand gesture classification using a CNN model
- Robust landmark normalization (translation and scaling)
- Support for left and right hand gestures
- Special UNKNOWN class for ambiguous or invalid gestures
- Temporal smoothing using a sliding window
- Automatic generation of evaluation metrics and plots

---

## Supported Gestures

The following static gestures are supported (for both left and right hand where applicable):

- FIST
- OPEN
- PEACE
- OK
- LIKE
- DISLIKE
- POINT
- MIDDLE
- UNKNOWN

---

## Model Architecture

The CNN model operates on hand landmarks represented as a sequence of 21 points with 3D coordinates (x, y, z).

Architecture overview:
- Input: (21, 3) landmark coordinates
- Conv1D layers for spatial feature extraction
- Batch Normalization for training stability
- Global Average Pooling
- Fully connected layer with dropout and L2 regularization
- Softmax output layer for gesture classification

---

## Project Structure

project_root/

│
├── data/

│   └── data.csv   


│
├── models/

│   ├── gesture_cnn.h5  # Trained CNN model

│   └── gesture_cnn_labels.pkl # Class label mapping

│
├── reports/

│   ├── confusion_matrix.png # Normalized confusion matrix

│   ├── training_curves.png  # Accuracy and loss curves

│   └── classification_report.txt

│
├── src/

│   ├── collect_guided_cnn.py # Guided dataset collection

│   ├── train_cnn.py     # Model training

│   ├── evaluate_cnn.py     # Model evaluation

│   └── realtime_cnn.py    # Real-time gesture recognition 

│

├── README.md

└── requirements.txt


---

## Evaluation

The model is evaluated using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The trained model achieves high overall accuracy and shows minimal confusion between gesture classes.
Training curves indicate stable learning behavior without significant overfitting.

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Collect dataset:
   python src/collect_guided_cnn.py

3. Train CNN model:
   python src/train_cnn.py

4. Evaluate model:
   python src/evaluate_cnn.py

5. Run real-time gesture recognition:
   python src/realtime_cnn.py

---

## Notes

- A webcam is required for dataset collection and real-time inference.
- Consistent lighting conditions improve recognition accuracy.
- The UNKNOWN class increases robustness by rejecting uncertain predictions.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
