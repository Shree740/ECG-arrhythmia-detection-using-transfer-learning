🩺 CardioNet: ECG Arrhythmia Detection using Deep Learning
📌 Overview

This project implements an ECG arrhythmia classification system using deep transfer learning.
It leverages pretrained CNN architectures (VGG16, ResNet50, InceptionV3, MobileNetV2) to classify ECG images into different arrhythmia categories.
A hyperparameter search was conducted to evaluate optimizers, learning rates, and epochs, with VGG16 + Adam (lr=1e-4) achieving the best performance.

✨ Features

✅ Dataset: ECG images from Kaggle (evilspirit05/ecg-analysis
)

✅ Deep Learning Models: VGG16, ResNet50, InceptionV3, MobileNetV2

✅ Transfer Learning & Fine-Tuning

✅ Data Augmentation (rotation, zoom, shift, flip)

✅ Hyperparameter Search (optimizer, learning rate, epochs)

✅ Evaluation: Accuracy, F1-score, Confusion Matrix

✅ Custom Prediction: Detect arrhythmia from new ECG images

📂 Project Structure
├── data/                        # ECG dataset (train/test split)
├── notebooks/                   # Jupyter/Colab notebooks
├── models/                      # Saved trained models (.h5)
├── results/                     # Training results & plots
├── results.csv                  # Hyperparameter search results
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/CardioNet-ECG.git
cd CardioNet-ECG

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Download the dataset

Create a Kaggle account

Get your kaggle.json API key from [Kaggle > Account > API]

Place it inside ~/.kaggle/

Run:

kaggle datasets download -d evilspirit05/ecg-analysis -p ./data --unzip

🏋️‍♂️ Training

To train the best model (VGG16 + Adam, lr=0.0001):

python train.py


(Or run the provided Colab notebook for training on GPU.)

📊 Results

Best Model: VGG16 + Adam (lr=0.0001)

Accuracy: ~82%

F1-Score: ~81%

Confusion Matrix (Heatmap)

Training Curves

🔍 Inference: Predict Arrhythmia from New ECG Image
from tensorflow.keras.models import load_model
from utils import predict_ecg

model = load_model("models/VGG16_adam_0.0001.h5")
print(predict_ecg("sample_ecg.png", model, class_indices))

📌 Future Work

Deploy model as a Flask/Streamlit web app

Export to TensorFlow Lite for mobile/IoT deployment

Extend to multi-lead ECG signals

🙌 Acknowledgements

Dataset: Kaggle - ECG Analysis

Pretrained CNN models: TensorFlow/Keras Applications

📜 License

This project is licensed under the MIT License.
