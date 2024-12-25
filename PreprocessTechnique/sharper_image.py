
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import unsharp_mask
from skimage import io
from skimage.color import rgb2hsv, hsv2rgb


def plot_comparison(original, filtered, title_original, title_filtered, v_diff=None):
    plt.figure(figsize=(25, 15))

    num = 131

    plt.subplot(num)
    plt.imshow(original, cmap='gray')
    plt.title(title_original)
    plt.axis('off')

    if v_diff is not None:
        num += 1
        plt.subplot(num)
        plt.imshow(v_diff, cmap='gray')
        plt.title('V Difference')
        plt.axis('off')

    num += 1
    plt.subplot(num)
    plt.imshow(filtered, cmap='gray')
    plt.title(title_filtered)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def rgb_to_hsv(image):
    hsv_image = rgb2hsv(image)
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    # Normalize and sharpen the channels
    h_normalized = h / 360.0 if h.max() > 1 else h
    s_normalized = s / 255.0 if s.max() > 1 else s
    v_normalized = v / 255.0 if v.max() > 1 else v

    return h_normalized, s_normalized, v_normalized



def sharpen_hsv(image, plot=False):
    h, s, v = rgb_to_hsv(img)

    # ------- Call The Image Sharpening Function --------
    v_sharpened = unsharp_mask(v, radius=25, amount=2.0)

    # Reconstruct the HSV image and convert back to RGB
    result_image = np.stack((h, s, v_sharpened), axis=-1)
    result_image = hsv2rgb(result_image)
    result_image = (result_image * 255).astype(np.uint8)  # Convert to uint8 for saving

    # Construct output file path
    original_filename = os.path.basename(imgPath)  # Get the original image name
    output_path = os.path.join(output_directory, original_filename)
    io.imsave(output_path, result_image)

    if plot:
        difference = np.abs(v - v_sharpened)
        plot_comparison(image, result_image, "Original", "HSV Sharpened", difference)



# Specify the paths
output_directory = "./PreprocessTechnique/sharpened_images/"  # Specify your output directory here
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# # ------- Loop through all JPG files in the directory --------
img_directory = "../dataset/Dates/"
i=1
for imgPath in glob.glob(os.path.join(img_directory, "*.jpg")):
    img = io.imread(imgPath)
    filtered_img = apply_bilateral_filter(img, d=75, sigma_color=75, sigma_space=9)

    print(f"Processing ({i}/2827): {imgPath} with shape: {img.shape}")

    # Apply the sharpening filter
    sharpen_hsv(filtered_img, plot=True if ((i % 24 == 0) or (i <= 15)) else  False)

    i+=1


