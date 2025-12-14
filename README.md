# 🎭 Face Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**Face Emotion Recognition** is an end-to-end Deep Learning–based system capable of detecting and classifying human facial expressions in real time via webcam or from static images.

This project utilizes the **EfficientNetB0** architecture through *Transfer Learning* techniques to achieve an optimal balance between high accuracy and computational efficiency.

---

## ✨ Key Features

* **Modern Training Pipeline**
  Utilizes the latest TensorFlow/Keras with a *Fine-Tuning* strategy (freezing initial layers, then retraining final layers).

* **Real-time Detection**
  Detects facial emotions directly from a webcam feed.

* **Photo Detection**
  Captures an image from the webcam and predicts the detected face emotion.

* **Data Augmentation**
  Uses `RandomFlip`, `RandomRotation`, and `RandomZoom` to reduce overfitting.

* **Metric Visualization**
  Includes Accuracy/Loss plots and Confusion Matrix for evaluation.

---

## 🧠 Model Architecture

The model is built upon **EfficientNetB0** (pretrained on ImageNet) with a custom classification head:

1. **Global Average Pooling** – Flattens feature maps
2. **Batch Normalization** – Stabilizes training
3. **Dropout (0.4)** – Reduces overfitting
4. **Dense Layer (256 units, ReLU)**
5. **Output Layer (5 units, Softmax)** – Emotion probabilities

---

## 😄 Emotion Classes

The system is trained to recognize **5 emotion classes**:

* 😠 **Angry**
* 😨 **Fear**
* 😄 **Happy**
* 😢 **Sad**
* 😲 **Suprise**

---

## 🚀 How to Run

### 1️⃣ Prerequisites

Ensure you have **Python 3.10 or 3.11** installed.

```bash
# Clone this repository
git clone https://github.com/404-MuhammadReza/face-emotion-recognition.git
cd face-emotion-recognition
```

(Optional) Create a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

> ⚠️ **Important**
> Ensure `haarcascade_frontalface_default.xml` is present in the project root directory.
> You can download it from the official OpenCV repository.

---

### 3️⃣ Model Training

* Open and run **`app.ipynb`**
* Ensure `dataset/train` and `dataset/test` folders are organized by class
* The trained model will be saved as:

```text
best_emotion_model.keras
```

---

### 4️⃣ Testing (Inference)

#### 🎥 Real-time Video Mode

Run `video-recognition.ipynb`:

* Opens webcam
* Detects face
* Displays emotion label + confidence
* Press **`q`** to exit

#### 📸 Photo Mode

Run `photo-recognition.ipynb`:

* Countdown
* Captures photo from webcam
* Analyzes facial expression

---

## 📊 Model Performance

Evaluation on the **Test Set**:

* **Global Accuracy**: ~66%
* **Strengths**: High accuracy on *Happy* and *Suprise*
* **Challenges**: Confusion between *Fear* and *Sad* due to similar facial features
