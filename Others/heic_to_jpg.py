"""
heic_to_jpg.py

This module converts HEIC files to JPG format. It reads all HEIC files in a specified 
directory, loads them using the pyheif library, and saves them as JPG images. This 
conversion is essential for compatibility with various image processing libraries.
"""

import os
import pyheif
from PIL import Image

def convert_heic_to_jpg(directory):
    # Get a list of all .heic files in the directory
    heic_files = [f for f in os.listdir(directory) if f.lower().endswith('.heic')]
    
    # Loop through each .heic file and convert it to .jpg
    for filename in heic_files:
        # Load the HEIC file
        heif_file = pyheif.read(os.path.join(directory, filename))
        
        # Convert HEIC to an image object
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data, 
            "raw", 
            heif_file.mode, 
            heif_file.stride
        )
        
        # Define new filename with .jpg extension
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        new_file_path = os.path.join(directory, new_filename)
        
        # Save the image as a .jpg file
        image.save(new_file_path, "JPEG")
        print(f"Converted {filename} to {new_filename}")

    print(f"Converted {len(heic_files)} .heic files to .jpg in the folder {directory}.")

# Specify the folder where the .heic files are located
# folder_path = "/home/mohammed/Downloads/Telegram Desktop/13/13/"  # Change this to your folder path

# convert_heic_to_jpg(folder_path)


def convert_png_to_jpg(input_directory, output_directory):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get a list of all .png files in the directory
    png_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.png')]
    
    # Loop through each .png file and convert it to .jpg
    for filename in png_files:
        # Load the PNG file
        input_path = os.path.join(input_directory, filename)
        image = Image.open(input_path)
        
        # Convert to RGB if image is in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Define new filename with .jpg extension
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_directory, new_filename)
        
        # Save the image as a .jpg file
        image.save(output_path, "JPEG")
        print(f"Converted {filename} to {new_filename}")
    
    print(f"Converted {len(png_files)} .png files to .jpg from {input_directory} to {output_directory}.")


convert_png_to_jpg(input_directory="static/processed_results2/", output_directory="static/processed_results4/")