# Image Background Remover Python Script With U2Net

This project utilizes the pretrained U2NET model for background removal. The code for background removal is based on the [Image-Background-Remover-Python](https://github.com/hassancs91/Image-Background-Remover-Python) repository. For more details on how the model was implemented, please refer to the original repository.

## Overview
The Image Background Remover is a program designed to remove backgrounds from images using advanced techniques such as deep learning and image processing. This code was specifically used for cleaning, removing the background and processing date images. It was used during the Medina Hackathon 2023 for enhancing our dataset. The application leverages a pre-trained U2NET model for effective background removal and includes various scripts for processing images, renaming files, and analyzing pixel values.

## Usage
To use the image background remover, their is required structure for the directories for the program to work. The code and model inside the `Image Denoising` directory is independant of the files and directories outside of it. You can download the `Image Denoising` by itself to run this code individually.
1. Make sure the `Image Denoising` directory is downloaded and saved on your device
2. Install the required dependencies: `pip install -r requirements.txt`
3. You'll need to download the actual pre-trained model and save it inside the `saved_models/u2net/` directory. Download the model from this [Google Drive Folder](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)
4. You might need to fix the file paths used in the `app.py` file to allow the code to run on your own local machine
5. Run the app.py script: `python app.py`
