# 🩺 CNN Diagnostics — Pneumonia Detection from Chest X-Rays

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify chest X-ray images as **Normal** or **Pneumonia**, using the classic Kaggle Chest X-Ray Pneumonia dataset.

## 📌 Overview

Medical image diagnosis is time-consuming and requires expert radiologists. This project explores whether a relatively lightweight CNN can learn to distinguish pneumonia-affected lungs from healthy ones directly from chest X-ray images, as a proof-of-concept for AI-assisted diagnostic support.

## 🗂️ Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Splits:** Train / Validation / Test (pre-split in the dataset)
- Images are loaded via `ImageDataGenerator.flow_from_directory`, resized to **150×150**, and batched at **32**.

## 🧠 Model Architecture

A sequential CNN:

```
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Conv2D(128, 3x3, relu) → MaxPooling2D
Flatten
Dense(128, relu) → Dropout(0.5)
Dense(1, sigmoid)
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam (learning rate = 0.0001)
- **Epochs:** 30
- **Metric:** Accuracy

## 🔄 Data Pipeline & Augmentation

To reduce overfitting and improve generalization, training images are augmented on the fly:

- Rescaling (1/255)
- Rotation (±30°)
- Width/height shift (±20%)
- Shear & zoom (20%)
- Horizontal flip
- `nearest` fill mode

Validation and test images are only rescaled (no augmentation), to evaluate on realistic, unaltered inputs.

## 📊 Results

| Metric | Value |
|---|---|
| Test Loss | 0.276 |
| Test Accuracy | ~90% |

The notebook also generates:
- A class-distribution bar chart across train/val/test splits (EDA)
- A classification report (precision/recall/F1 per class)
- A confusion matrix heatmap (Normal vs. Pneumonia)

> **Note:** There's a noticeable gap between the `model.evaluate()` accuracy (~90%) and the `classification_report` computed over the full test set (~54%). This points to a mismatch between the batched evaluation subset and the full prediction pass (e.g. generator shuffling/step-count truncation) — a good next step is to set `shuffle=False` on the test generator and evaluate over the *entire* test set for a fully consistent number.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (classification report, confusion matrix)

## 🚀 Getting Started

1. Download the dataset from Kaggle and place it so the following paths resolve:
   ```
   chest_xray/train/{NORMAL,PNEUMONIA}
   chest_xray/val/{NORMAL,PNEUMONIA}
   chest_xray/test/{NORMAL,PNEUMONIA}
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the notebook `cnn-diagnostics-medical-image.ipynb` end-to-end (or via `jupyter nbconvert --execute`).

## 📈 Future Improvements

- Fix test-set evaluation to run over all samples with `shuffle=False` for a trustworthy metric
- Add early stopping / learning-rate scheduling to reduce training time and overfitting
- Try transfer learning (e.g. ResNet50, EfficientNet, DenseNet121) pretrained on ImageNet
- Address class imbalance (Pneumonia > Normal in this dataset) with class weighting or resampling
- Add Grad-CAM visualizations to highlight which regions of the X-ray drive predictions
- Track experiments (e.g. with MLflow or Weights & Biases)

## 👤 Author

Mohd Zaid
