
# 🖋️ Telugu Handwritten Word-Level OCR System

This repository contains the code and dataset for the mini-project titled **"Telugu Handwritten Word-Level OCR System"**. The project involves developing an OCR pipeline that recognizes handwritten Telugu words using a deep learning-based model. It includes preprocessing with CRAFT-based word segmentation and a recognition model built using Transformers (TrOCR) with Telugu script support.

The pipeline is designed to enable the training and deployment of a fully functional OCR model that handles Telugu word recognition in real-world scanned documents.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Technologies Used](#technologies-used)
- [Environment Setup](#environment-setup)
- [Training and Testing](#training-and-testing)
- [License](#license)
- [Authors](#authors)

---

## 🔍 Project Overview

Handwritten text recognition in Indian languages like Telugu poses unique challenges due to complex scripts, varied writing styles, and limited labeled data. This project addresses these issues through:

- 🧠 **Preprocessing Pipeline**: Degradation removal + CRAFT-based word cropping (executed in Google Colab).
- 🔤 **Recognition Model**: Fine-tuned TrOCR model with the `<2te>` token to support Telugu characters.
- 🧪 **Evaluation Interface**: Test pipeline using FastAPI or Python script.

---

## 🚀 Features

- ✅ Word-level cropping using CRAFT (Google Colab)
- ✅ Dataset split into `train`, `val`, `test` with tab-separated labels
- ✅ Transformer-based OCR model using TrOCR and HuggingFace
- ✅ Batch test script and FastAPI interface for deployment

---

## 🗂️ Dataset Structure

```
/dataset/
├── train/               # Training images
├── val/                 # Validation images
├── test/                # Testing images
├── train.txt            # image_name.jpg<TAB>label
├── val.txt
└── test.txt
```

> Each `.txt` file maps image names to the ground truth word using **tab-separated values**.

---

## 🛠️ Technologies Used

- **Language**: Python
- **OCR Model**: HuggingFace TrOCR (ViT + GPT2)
- **Preprocessing**: Google Colab, OpenCV, DEGAN, CRAFT, CROP
- **APIs**: FastAPI for deployment
- **Visualization**: PIL, Matplotlib
- **Others**: torch, transformers, uvicorn

---

## 🧪 Environment Setup

### Create Conda Environment:

```bash
conda create -n telugu_ocr python=3.10.16
conda activate telugu_ocr
```

### Install Requirements:

```bash
pip install -r requirements.txt
```

---

## 🔧 Training and Testing

### Training

Make sure the dataset and `.txt` files are correctly structured, then run:

```bash
python Run.py
```

### Testing (Batch)

```bash
python test.py
```
CER - 0.0083

> You can also use the FastAPI interface to upload and predict single images.

---

## 📄 License

MIT License. See `LICENSE` file for details.

---

## 👩‍💻 Authors

Developed by **Sravani Peta** as part of a mini project on Indian language OCR systems.
