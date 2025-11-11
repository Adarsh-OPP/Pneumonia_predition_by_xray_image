# 🩺 Pneumonia Detection from Chest X-Ray Images
### Using EfficientNetB0 + Transfer Learning

This project applies deep learning to classify chest X-ray images as **NORMAL** or **PNEUMONIA**. It uses **EfficientNetB0**, a modern and high-performance convolutional neural network, combined with a custom classification head optimized for medical imaging.

The goal is to build a clear, reliable, and well-documented medical AI pipeline.

---

## 📘 Project Overview

**Dataset:** Chest X-Ray Pneumonia (Kaggle)

**Classes:**
- ✅ NORMAL
- ⚠️ PNEUMONIA

This project includes:
- ✅ Complete dataset preparation and cleanup
- ✅ Moving 350 images from test → train to rebalance data
- ✅ Medically-safe data augmentation
- ✅ Transfer learning using EfficientNetB0
- ✅ Custom classification head (GAP → Dropout → Dense)
- ✅ Training + evaluation pipeline
- ✅ Exported model in `.keras` format

Final performance: **~90% accuracy**, with potential for improvement via fine-tuning.

---

## 📂 Repository Contents

### 📄 pneumonia-process.ipynb
Handles all data preparation:
- Splits and reorganizes the dataset
- Applies augmentation
- Builds tf.data pipelines with caching + prefetching
- Prepares balanced train/validation/test directories

### 📄 pneumonia-prediction.ipynb
Contains the full model workflow:
- Loads EfficientNetB0 (ImageNet pretrained)
- Adds a custom classification head
- Trains and validates the model
- Evaluates performance
- Runs predictions on sample X-ray images

### 🧠 pneumonia_efficientnet_tf.keras
The trained model including:
- EfficientNetB0 backbone
- Custom classification layers
- All learned weights (ready for inference or fine-tuning)

---

## 🧱 Model Architecture

**Architecture Overview:**
- EfficientNetB0 backbone (feature extractor)
- Global Average Pooling
- Dropout (regularization)
- Dense softmax output layer (two-class classification)

**Benefits:**
- Lightweight and efficient
- Strong generalization
- Suitable for medical image analysis

---

## 📊 Performance Summary

The model achieves ~90% accuracy due to:
- Strong EfficientNet feature extraction
- Balanced dataset after preprocessing
- Carefully selected augmentations
- Clean train/validation/test structure

**Further improvements possible:**
- Fine-tuning deeper EfficientNet layers
- More augmentation
- Larger classification head
- Hyperparameter tuning

---

## 📁 Files Included
- pneumonia-process.ipynb
- pneumonia-prediction.ipynb
- pneumonia_efficientnet_tf.keras

---

## 🏅 Acknowledgements
Dataset: Chest X-Ray Pneumonia (Kaggle)  
Backbone Model: EfficientNetB0
