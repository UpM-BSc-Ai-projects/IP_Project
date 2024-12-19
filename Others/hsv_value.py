"""
hsv_value.py

This module analyzes the HSV values of an image to identify dark areas. It reads an 
image, converts it to the HSV color space, and creates a mask for dark regions. 
This analysis is useful for isolating specific features in the image for further 
processing.
"""

import cv2
import numpy as np

def analyze_hsv(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for dark areas (adjust these values if needed)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])  # Increased value range to capture more of the date
    mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Apply the mask
    dark_areas = cv2.bitwise_and(hsv, hsv, mask=mask)

    # Get non-zero pixels (i.e., the dark areas)
    non_zero = dark_areas[np.nonzero(dark_areas)]

    if non_zero.size == 0:
        print("No dark areas found in the image.")
        return

    # Calculate statistics
    min_vals = np.min(non_zero, axis=0)
    max_vals = np.max(non_zero, axis=0)
    mean_vals = np.mean(non_zero, axis=0)

    print(f"HSV Analysis for dark areas:")
    print(f"Minimum values: H={min_vals[0]}, S={min_vals[1]}, V={min_vals[2]}")
    print(f"Maximum values: H={max_vals[0]}, S={max_vals[1]}, V={max_vals[2]}")
    print(f"Mean values: H={mean_vals[0]:.2f}, S={mean_vals[1]:.2f}, V={mean_vals[2]:.2f}")

# Usage
image_path = '/home/mohammed/Documents/ImageBackgroundRemover/static/results/JD2007_3.png'

analyze_hsv(image_path)