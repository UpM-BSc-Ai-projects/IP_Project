"""
heic_rename.py

This script renames all HEIC files in a specified directory to a standardized format. 
It retrieves all HEIC files, sorts them, and renames them sequentially to ensure 
consistent naming for easier management and processing.
"""

import os

def rename_heic_files(directory):
    # Get a list of all .heic files in the directory
    heic_files = [f for f in os.listdir(directory) if f.lower().endswith('.heic')]
    
    # Sort the files to ensure consistent ordering
    heic_files.sort()
    
    # Loop through each file and rename it
    for i, filename in enumerate(heic_files, start=1):
        # Define new filename
        new_filename = f"JT2020_{i}.heic"
        
        # Get the full file path
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")

    print(f"Renamed {len(heic_files)} .heic files in the folder {directory}.")

# Specify the folder where the .heic files are located
folder_path = r'/home/mohammed/Downloads/Telegram Desktop/20/20/'  # Change this to your folder path

rename_heic_files(folder_path)
