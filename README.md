# Image Upscaling – Deep Learning Image Super Resolution

This project focuses on **enhancing low-resolution images into high-resolution images** using **deep learning techniques**, mainly **RRDN (Residual-in-Residual Dense Network)**.  
The goal is to improve image clarity and sharpness **without changing original colors or textures**.

---

## Project Overview

Image Super-Resolution is the process of converting a low-resolution image into a higher-resolution version while preserving visual quality.

In this project:
- RRDN model is trained **from scratch**
- Low-resolution images are enhanced to HD quality
- Image quality is evaluated using **PSNR** and **SSIM**
- Output images maintain original color consistency

---

## Technologies Used

- Python  
- PyTorch  
- NumPy  
- OpenCV  
- Deep Learning (CNNs)  
- RRDN Architecture  

---

## 📁 Project Structure

```
enhancing_image/
│
├── dataset/                 # Training and validation images
├── models/                  # Saved model weights
├── RRDN_arch.py             # RRDN model architecture
├── train_rrdn.py            # Training script
├── final_enhance.py         # Image enhancement script
├── requirements.txt         # Required libraries
└── README.md                # Project documentation
```

---

## Key Features

- ✔️ RRDN-based Super Resolution  
- ✔️ Trained on custom datasets  
- ✔️ No color distortion in output images  
- ✔️ Supports custom input images  
- ✔️ PSNR & SSIM evaluation metrics  
- ✔️ Modular and clean code structure  

---

## Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SvvNeeraj/enhancing_image.git
cd enhancing_image
```

### 2️⃣ Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Training the RRDN Model

To train the model on your dataset:

```bash
python train_rrdn.py --data_dir dataset/
```

- Trains RRDN from scratch  
- Saves trained model weights  
- Uses high-resolution and low-resolution image pairs  

---

## Image Enhancement (Inference)

Enhance a low-resolution image:

```bash
python final_enhance.py --input_image path/to/image.jpg
```

This will:
- Generate a high-resolution image  
- Preserve original colors  
- Display PSNR and SSIM values  

---

## 📊 Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)** – Measures reconstruction quality  
- **SSIM (Structural Similarity Index)** – Measures perceptual similarity  

Higher values indicate better image quality.

---

## Project Objectives

- Improve low-resolution images using deep learning  
- Implement RRDN architecture practically  
- Maintain color accuracy during enhancement  
- Build a complete training + inference pipeline  

---

This Deep learning project performs image upscaling (super-resolution) using a custom-trained RRDN (Residual-in-Residual Dense Network) model. The goal is to enhance low-resolution images by training on a dataset and generating high-quality, high-resolution outputs while preserving color and detail. Repository files RRDN_arch.py Contains the architecture definition for the RRDN model. This file builds the deep learning model used for both training and inference.

train_rrdn.py Script to train the RRDN model using a custom dataset of low-resolution and high-resolution image pairs.
Output: A trained model file (e.g rrdn_model.pth).
where this pth file is used main code to enhance image

final_enhance.py Contains the final code to upscale the image using streamlit interface.There two metrics measure values of PSNR and SSIM which are used to know the actually the image upscaled or not 

Take Image Deblurring datasets folder RealBlur_R from kaggle to train rrdn model.
