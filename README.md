# 🩺 Pneumonia Detection from Chest X-Ray Images  
### Using EfficientNetB0 + Transfer Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![GPU](https://img.shields.io/badge/Accelerator-GPU%20(P100%2FT4)-green?logo=nvidia)
![Model](https://img.shields.io/badge/Model-.keras-purple?logo=keras)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Last Commit](https://img.shields.io/github/last-commit/Adarsh-OPP/Pneumonia_predition_by_xray_image)
![Stars](https://img.shields.io/github/stars/Adarsh-OPP/Pneumonia_predition_by_xray_image?style=social)
![Forks](https://img.shields.io/github/forks/Adarsh-OPP/Pneumonia_predition_by_xray_image?style=social)

---

This project applies deep learning to classify chest X-ray images as **NORMAL** or **PNEUMONIA** using **EfficientNetB0**.  
The model combines a modern pretrained backbone with a custom classification head optimized for medical imaging.

The goal is to build a clear, reliable, and well-documented medical AI pipeline.

---

## 📘 Project Overview

**Dataset:** Chest X-Ray Pneumonia (Kaggle)

**Classes:**
- ✅ NORMAL  
- ⚠️ PNEUMONIA  

Key Features:
- ✅ Complete dataset preparation and cleanup  
- ✅ 350 images moved from test → train for balancing  
- ✅ Medically-safe augmentation  
- ✅ EfficientNetB0 transfer learning  
- ✅ Custom classification head (GAP → Dropout → Dense)  
- ✅ Full training + evaluation workflow  
- ✅ Exported model in `.keras` format  

Final performance: **~90% accuracy**, with potential for improvement via fine-tuning.

---

## 📂 Repository Contents

### 📄 pneumonia-process.ipynb  
Handles all data preprocessing:
- Directory restructuring  
- Data balancing  
- Augmentation  
- Built optimized `tf.data` pipelines  

### 📄 pneumonia-prediction.ipynb  
Full training pipeline:
- Load EfficientNetB0 backbone  
- Attach classification head  
- Train, validate, and evaluate  
- Predict on new X-ray images  

### 🧠 pneumonia_efficientnet_tf.keras  
Trained model including:
- EfficientNetB0 backbone  
- Custom layers  
- All trained weights  

---

## 🧱 Model Architecture

**Architecture:**
- EfficientNetB0 backbone  
- Global Average Pooling  
- Dropout  
- Dense softmax classifier  

**Benefits:**
- High accuracy  
- Lightweight and fast  
- Strong generalization  
- Well-suited for medical imaging  

---

## 📊 Performance Summary

Achieved **~90% accuracy** due to:
- Strong feature extraction  
- Balanced dataset  
- Effective augmentation  
- Clean train/val/test split  

**Future improvements:**
- Fine-tune deeper EfficientNet layers  
- Use additional augmentation  
- Add ROC-AUC, confusion matrix  
- Optimize hyperparameters  

---

## 📁 Files Included

- pneumonia-process.ipynb  
- pneumonia-prediction.ipynb  
- pneumonia_efficientnet_tf.keras  

---

## 🏅 Acknowledgements

Dataset: Chest X-Ray Pneumonia (Kaggle)  
Backbone Model: EfficientNetB0  
