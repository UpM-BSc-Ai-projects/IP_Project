# Date Image Preprocessing & 3D Image Reconstruction

## **Table of Contents**
1. [Image Preprocessing](#image-preprocessing)
    - [Image Background Remover With U2Net](#image-background-remover-python-script-with-u2net)
        - [Overview](#overview)
        - [Usage](#usage)

2. [3D Image Reconstruction Using InstantMesh](#3d-image-reconstruction-using-instantmesh)
    - [Examples](#3d-examples)
    - [Model Architecture](#model-architecture)
    - [Dependencies and Installation](#dependencies-and-installation)

3. [Others](#others)
    - [Nutritional Value Prediction of Dates Using Machine Learning and Deep Learning](#nutritional-value-prediction-of-dates-using-machine-learning-and-deep-learning)
        - [Objective](#objective)
        - [Dataset](#dataset)
        - [Usage](#nutritional-usage)


---

# Image Preprocessing

## Image Background Remover Python Script With U2Net

This project utilizes the pretrained U2NET model for background removal. The code for background removal is based on the [Image-Background-Remover-Python](https://github.com/hassancs91/Image-Background-Remover-Python) repository. For more details on how the model was implemented, please refer to the original repository.

### Overview

The Image Background Remover is a program designed to remove backgrounds from images using advanced techniques such as deep learning and image processing. This code was specifically used for cleaning, removing the background and processing date images. It was used during the Medina Hackathon 2023 for enhancing our dataset. The application leverages a pre-trained U2NET model for effective background removal and includes various scripts for processing images, renaming files, and analyzing pixel values.

### Usage

To use the image background remover, their is required structure for the directories for the program to work. The code and model inside the `Image Denoising` directory is independant of the files and directories outside of it. You can download the `Image Denoising` by itself to run this code individually.

1. Make sure the `Image Denoising` directory is downloaded and saved on your device
2. Install the required dependencies: `pip install -r requirements.txt`
3. You'll need to download the actual pre-trained model `u2net.pth` and save it inside the `saved_models/u2net/` directory. Download the model from this [Google Drive Folder](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)
4. You might need to fix the file paths used in the `__init__.py` file to allow the code to run on your own local machine
5. Run the app.py script: `python __init__.py`


---

# 3D Image Reconstruction Using InstantMesh

This repo is the derived implementation of InstantMesh. For original repo, visit https://github.com/TencentARC/InstantMesh

For a Google Colab demo, visit [![Button Text](https://img.shields.io/badge/Button-Click%20Here-blue)](https://colab.research.google.com/drive/1spbyRA6ZNWDsZU1ZHt_-w_a4aeqZF9KU?usp=sharing)

## 3D Examples

![Demo](https://github.com/UpM-BSc-Ai-projects/IP_Project/blob/main/CloneInstantMesh/BD_3D_example.gif)
![Demo](https://github.com/UpM-BSc-Ai-projects/IP_Project/blob/main/CloneInstantMesh/JD_3D_example.gif)
![Demo](https://github.com/UpM-BSc-Ai-projects/IP_Project/blob/main/CloneInstantMesh/KD_3D_example.gif)

## Model Architecture

![Logo](https://github.com/UpM-BSc-Ai-projects/IP_Project/blob/main/CloneInstantMesh/model_architecture.png)

## Dependencies and Installation

We recommend using Python>=3.10, PyTorch>=2.1.0, and CUDA>=12.1.

```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt
```

# Others

---

## Nutritional Value Prediction of Dates Using Machine Learning and Deep Learning

This project aims to predict the nutritional values of date fruits based on their images using machine learning (ML) and deep learning (DL). The solution leverages Convolutional Neural Networks (CNNs) and pretrained architectures to deliver accurate and efficient predictions.

### Objective

The goal is to estimate the following nutritional parameters of date fruits:

- Calories
- Carbohydrates
- Proteins
- Total Fat
- Glucose
- Cholesterol
- Vitamins
- Water Content
- Energy

The predictions are made directly from date images, reducing the need for manual or laboratory-based testing.

### Dataset

1. **Images:**
   - Images of 8 types of dates, stored in separate folders.
   - Each image corresponds to a specific date variety.
2. **Labels:**
   - `nutritional_values.csv`: A file containing nutritional values for each image.

#### Preprocessing

- Images resized to:
  - **28x28 (CNNs)** for simplicity.
  - **224x224 (Pretrained Models)** for feature extraction.
- Normalized pixel values to range `[0, 1]`.
- Split into **80% training** and **20% testing** datasets.

### Nutritional Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/UpM-BSc-Ai-projects/IP_Project.git
   cd nutritional-value-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python train_model.py
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Upload a date image to predict its nutritional values.

For more detailed information about the model used and performance, refer to the [full README](https://github.com/UpM-BSc-Ai-projects/IP_Project/tree/main/Nutritional%20Values%20predictor#readme).


