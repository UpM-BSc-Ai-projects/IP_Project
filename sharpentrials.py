import os
import random
from skimage import io, filters, img_as_float
from skimage.filters import unsharp_mask
from skimage.color import rgb2gray
import numpy as np

import matplotlib.pyplot as plt

# Directory containing the images, extract it from the zip file
image_dir = 'Dates/'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Randomly select 3 images
selected_images = random.sample(image_files, 3)

# Plotting the images
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
ax = axes.ravel()

def apply_high_pass_filter(image_gray):
        # Apply Fourier transform
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)

        # Create a high pass mask
        rows, cols = image_gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # Radius of the mask
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0

        # Apply the mask and inverse Fourier transform
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back

for i, image_file in enumerate(selected_images):
    # Read the image
    image = img_as_float(io.imread(os.path.join(image_dir, image_file)))

    # Apply unsharp mask with different parameters
    sharpened_unsharp = unsharp_mask(image, radius=1, amount=1)
    # Apply Fourier transform and contouring
    image_gray = rgb2gray(image)

    img_back = apply_high_pass_filter(image_gray)

    contours = filters.sobel(image_gray)

    # Plot original and sharpened images with different parameters
    ax[i*4].imshow(image)
    ax[i*4].set_title('Original Image')

    ax[i*4 + 1].imshow(img_back, cmap='gray')
    ax[i*4 + 1].set_title('Fourier Transform')

    ax[i*4 + 2].imshow(contours, cmap='gray')
    ax[i*4 + 2].set_title('Contours')

    ax[i*4 + 3].imshow(sharpened_unsharp, cmap='gray')
    ax[i*4 + 3].set_title('Unsharp Mask')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()