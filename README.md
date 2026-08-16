# Handwritten Character Recognition using CNN

## 📌 Project Overview

Handwritten Character Recognition is a deep learning project that identifies handwritten digits from images using a **Convolutional Neural Network (CNN)**.

This project uses the **MNIST handwritten digit dataset** and provides a simple graphical interface where users can draw a digit and get the predicted result along with the confidence score.

## 🎯 Objective

The main objective of this project is to develop an AI-based system that can recognize handwritten digits using image processing and deep learning techniques.

The project can be extended in the future to recognize handwritten alphabets, words, and complete sentences.

## 🚀 Features

* Handwritten digit recognition from 0–9
* CNN-based deep learning model
* MNIST dataset
* Image preprocessing and normalization
* Real-time prediction
* Prediction confidence score
* Interactive Tkinter drawing interface
* Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
* Extendable to alphabet and word recognition

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pillow (PIL)**
* **Matplotlib**
* **Scikit-learn**
* **Tkinter**
* **CNN (Convolutional Neural Network)**

## 📂 Project Structure

```text
handwritten_character_recognition/
│
├── train_model.py       # Train and evaluate the CNN model
├── app.py               # Graphical interface for digit recognition
├── requirements.txt     # Required Python libraries
├── mnist_cnn.keras      # Trained CNN model
└── README.md            # Project documentation
```

## 🧠 How It Works

The system follows these steps:

```text
MNIST Dataset
      ↓
Image Preprocessing
      ↓
CNN Model Training
      ↓
Model Evaluation
      ↓
Save Trained Model
      ↓
User Draws Digit
      ↓
Image Resizing & Normalization
      ↓
CNN Prediction
      ↓
Predicted Digit + Confidence
```

## 📊 Dataset

This project uses the **MNIST dataset**, which contains handwritten digits from 0 to 9.

* Training images: 60,000
* Testing images: 10,000
* Image size: 28 × 28 pixels
* Classes: 10 (0–9)

The dataset is automatically downloaded by TensorFlow when `train_model.py` is executed for the first time.

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/handwritten-character-recognition.git
```

### 2. Open the project folder

```bash
cd handwritten-character-recognition
```

### 3. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

If `python` does not work, try:

```bash
py -m pip install -r requirements.txt
```

## ▶️ How to Run

### Step 1: Train the model

Run:

```bash
python train_model.py
```

This will:

* Download the MNIST dataset
* Preprocess the images
* Build the CNN model
* Train the model
* Evaluate the model
* Display the classification report
* Display the confusion matrix
* Save the trained model as `mnist_cnn.keras`

### Step 2: Start the application

After training is completed, run:

```bash
python app.py
```

A graphical window will open.

Draw a digit using your mouse and click **Predict**.

The application will display:

```text
Predicted Digit: 7
Confidence: 98.XX%
```

## 🧩 CNN Architecture

The CNN model consists of:

```text
Input Layer
    ↓
Conv2D (32 filters)
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters)
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (128 neurons)
    ↓
Dropout
    ↓
Output Layer (10 classes)
```

The final output contains probabilities for digits **0–9**.

## 📈 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The training program also displays training and validation accuracy.

## 🖥️ Application

The project includes a Tkinter-based graphical user interface.

Users can:

1. Draw a handwritten digit.
2. Click **Predict**.
3. View the predicted digit.
4. View the prediction confidence.
5. Clear the drawing and try another digit.

## 🔮 Future Enhancements

This project can be extended with:

* Handwritten alphabet recognition using **EMNIST**
* A–Z character recognition
* Full word recognition
* Sentence recognition
* CRNN (CNN + RNN) architecture
* Transformer-based handwriting recognition
* Web-based interface using Flask
* Mobile application
* Handwriting-to-text conversion
* Support for multiple languages

## 📚 Learning Outcomes

Through this project, we learn:

* Image preprocessing
* Deep learning fundamentals
* Convolutional Neural Networks
* Model training and evaluation
* Classification
* Computer vision
* GUI development using Tkinter
* Deploying a trained machine learning model

## 👩‍💻 Author

**Chandana M M**

Electronics and Communication Engineering Student

GitHub: https://github.com/chandana1802

## ⭐ Acknowledgement

This project uses the MNIST handwritten digit dataset and TensorFlow/Keras for implementing the CNN-based recognition model.

---

⭐ If you find this project useful, consider giving the repository a star!
