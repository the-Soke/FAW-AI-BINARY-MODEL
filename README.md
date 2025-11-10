
# 🐛 Fall Armyworm (FAW) Early Detection System

A supervised machine learning project for detecting **Fall Armyworm (FAW)** from image data.
This project implements a **binary classification model** that identifies whether an image contains FAW presence or not — supporting early pest detection and management in agriculture.

---

## 📘 Project Overview

The **Fall Armyworm (Spodoptera frugiperda)** is a destructive agricultural pest responsible for massive crop losses worldwide, especially in maize.
This project focuses on **automating early detection** of FAW using a supervised deep learning approach.

The system classifies input images into two categories:

- ✅ **FAW** – Fall Armyworm or infestation signs detected
- 🌾 **No-FAW** – Healthy crop, no infestation

The final trained model is exported in the **ONNX** format for flexible deployment across platforms such as mobile apps, web dashboards, or drone systems.

---

## 🚀 Getting Started

To set up the project and organize the files, follow these steps:

1.  **Clone the repository:**
    ```bash
    !git clone https://github.com/the-Soke/FAW-AI-BINARY-MODEL.git
    %cd FAW-AI-BINARY-MODEL
    ```

2.  **Organize project files:**
    The following commands will set up the required directory structure:

    *   **Move `model.pt` to the `models` directory and `model.onnx` to the `ONNX` directory:**
        ```bash
        !mkdir -p models
        !mv model.pt models/
        !mkdir -p ONNX
        !mv model.onnx ONNX/
        ```
    *   **Move `armyWorm` and `non_FAW` folders to the `Dataset` directory:**
        ```bash
        !mkdir -p Dataset
        !mv armyWorm Dataset/
        !mv non_FAW Dataset/
        ```
    *   **Move notebook files (`ONNX_export.ipynb`, `PIPELINE.ipynb`, `PIPELINE_METRICS.ipynb`) to the `Notebooks` directory:**
        ```bash
        !mkdir -p Notebooks
        !mv ONNX_export.ipynb Notebooks/
        !mv PIPELINE.ipynb Notebooks/
        !mv PIPELINE_METRICS.ipynb Notebooks/
        ```

---

## 🎯 Objectives

- Build a **supervised image classification model** for FAW detection.
- Apply **data preprocessing and augmentation** to enhance performance.
- Evaluate model performance using **Accuracy, Precision, Recall, and F1-Score**.
- Export the final model to **ONNX format** for real-world deployment.
- Maintain **reproducibility and clarity** through clean, well-documented code.

---

## 🧠 Model Workflow

1.  **Data Loading**
    - Load the custom FAW dataset (located in `Dataset/armyWorm` and `Dataset/non_FAW`).
    - Split into training, validation, and test sets.

2.  **Data Preprocessing**
    - Resize, normalize, and augment images (rotation, zoom, flip, etc.).

3.  **Model Training**
    - Train a CNN-based image classifier using TensorFlow/Keras or PyTorch.
    - Output labels:
        - `1` → FAW
        - `0` → No-FAW

4.  **Model Evaluation**
    - Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    - Optimize hyperparameters to improve generalization.

5.  **Model Export**
    - Save the best-performing model as `model.onnx` to the `ONNX/` directory for deployment.

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Environment | Google Colab |
| Programming | Python 3 |
| Core ML | TensorFlow / Keras / PyTorch |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Export | ONNX |

---

## 📊 Expected Output

- Trained model: `ONNX/model.onnx`
- Evaluation report (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix visualization
- Example predictions on test images

---

## 🚀 Future Improvements

- Extend model to **multi-class detection** (e.g., larva, moth, damage).
- Integrate with **Telegram bot** or **drone scouting system** for real-time detection.
- Optimize for **edge deployment** using TensorFlow Lite or ONNX Runtime Mobile.

---

## 🧾 Acknowledgment

This project is part of the **AI Capstone Project: Fall Armyworm (FAW) Early Detection System**,
developed as part of the **AI Bootcamp** initiative.

---
