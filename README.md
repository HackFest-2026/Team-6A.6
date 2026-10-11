# 🫁 AI-Powered Pneumonia Detection using Deep Learning & EfficientNet

> **A Biotechnology Hackathon Project** designed to automate medical image classification and assist healthcare professionals in preliminary chest X-ray diagnostics.

---

## 📌 Executive Summary
Pneumonia is a life-threatening respiratory infection that causes inflammation in the air sacs of one or both lungs. Early and accurate diagnosis is critical for effective medical intervention. However, manual interpretation of chest X-ray images requires specialized expertise and can be time-consuming, especially in high-volume or resource-limited medical settings.

This project introduces an automated end-to-end deep learning framework utilizing the **EfficientNet** architecture to detect pneumonia from chest X-ray images with high accuracy, speed, and precision.

---

## 🎯 Key Objectives & Features
* **Automated Image Analysis:** Fast evaluation of chest X-ray scans to classify them into **Normal** vs. **Pneumonia**.
* **Transfer Learning Integration:** Utilizes pre-trained state-of-the-art **EfficientNet** deep learning models to achieve superior classification accuracy with reduced training overhead.
* **Biomedical Relevance:** Specifically tailored for biotechnology and healthcare applications, serving as a decision-support tool for clinicians.
* **Reproducible Workflow:** Full pipeline structured within a Google Colab environment for easy replication and evaluation.

---

## 📊 Dataset Overview
* **Dataset Name:** RSNA / Chest X-Ray Pneumonia Dataset
* **Modality:** Anterior-Posterior (AP) & Posterior-Anterior (PA) Chest Radiographs (X-Rays).
* **Classes:** 
  1. `Normal` — Clear lung fields without signs of focal opacification.
  2. `Pneumonia` — Manifesting focal or diffuse consolidation and opacity.
* **Preprocessing Pipeline:**
  * Resizing images to standardized input dimensions (e.g., $224 \times 224$ pixels).
  * Normalization and pixel intensity scaling.
  * Data augmentation techniques (rotation, brightness adjustments, flipping) to prevent overfitting.

---

## 🛠️ Technology Stack & Libraries
* **Programming Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras (or PyTorch)
* **Pretrained Model Architecture:** EfficientNet (B0/B3)
* **Data Processing & Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Image Processing:** OpenCV (`cv2`), PIL (Pillow)
* **Environment:** Google Colab / Jupyter Notebook

---

## 🔬 Model Architecture & Training Methodology
1. **Backbone Feature Extractor:** Leveraged EfficientNet pre-trained on ImageNet weights to extract high-level feature representations from X-ray scans.
2. **Custom Classification Head:** Added Global Average Pooling (GAP), Dropout layers for regularization, and Dense layers with Sigmoid/Softmax activation.
3. **Loss Function & Optimizer:** 
   * Loss Function: Binary Cross-Entropy
   * Optimizer: Adam / RMSprop with dynamic learning rate reduction.
4. **Evaluation Metrics:** Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC Curve.

---

## 📂 Project Structure
```text
├── pneumonia_hackathon.ipynb   # Main Google Colab notebook (Data preprocessing, model training & testing)
├── RSNA dataset x ray images.zip # Sample/Full dataset zip archive
├── README.md                   # Detailed project documentation
└── requirements.txt            # Python dependencies and library versions
