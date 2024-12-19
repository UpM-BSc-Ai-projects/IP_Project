
# Performs edge detection and noise removal on images using the Canny edge detection algorithm and watershed segmentation. 
# 
# Process:
# 1. **Color Conversion**: The function first converts the input image from RGBA to RGB format and then to grayscale.
# 
# 2. **Canny Edge Detection**: The Canny edge detection algorithm is applied to the grayscale image to identify edges.
# 
# 3. **Binary Filling**: The detected edges are filled using binary hole filling techniques to create a more complete representation of the edges.
# 
# 4. **Elevation Map Calculation**: An elevation map is generated using the Sobel filter applied to the grayscale image. This map helps in understanding the gradient of the image, which is useful for segmentation.
# 
# 5. **Morphological Dilation**: The image undergoes morphological dilation to enhance the features, making regions more distinct. This step helps in improving the results of subsequent segmentation.
# 
# 6. **Marker Creation for Watershed Segmentation**: Markers are created based on the dilated image. Pixels below a certain intensity are marked as background (2), while those above a higher threshold are marked as foreground (1).
# 
# 7. **Watershed Segmentation**: The watershed algorithm is applied to the elevation map using the markers to segment the image into distinct regions. This technique is useful for separating overlapping objects.
# 
# 8. **Fill Holes in Segmentation**: Any holes in the segmented regions are filled to create a binary mask that represents the segmented areas.
# 
# 9. **Labeling Segmented Regions**: The labeled regions are overlaid onto the original image for visualization purposes.
# 
# 10. **Padding Segmentation**: The segmentation mask is dilated by 20 pixels to create a padded contour. This helps in isolating the regions of interest more effectively.
# 
# 11. **Contour Analysis**: Each labeled contour is analyzed to check the percentage of pixel intensities below a specified threshold. If a contour meets the condition (i.e., a significant percentage of pixels are below the threshold), the corresponding area in the output image is set to white (indicating removal).
# 
# 12. **Output Visualization**: The final output image is displayed, showing the areas that have been processed based on the defined conditions.


import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from skimage.io import imread, imsave
from skimage import data, io, filters, color, feature, segmentation, measure, morphology
from skimage.exposure import  histogram
from skimage.morphology import dilation, disk
from skimage.color import label2rgb
import scipy.ndimage as nd


def edge_detect_and_noise_remover_colored(image, save_path=None):
    # Convert image to grayscale for processing
    grayscale_image = color.rgb2gray(image)
    
    # Step 1: Edge detection and hole filling on grayscale
    canny_edge = feature.canny(grayscale_image)
    fill_im = nd.binary_fill_holes(canny_edge)

    # Step 2: Elevation map and watershed segmentation on grayscale
    elevation_map = filters.sobel(grayscale_image)
    image_dilated = dilation(grayscale_image, disk(2))
    elevation_map = filters.sobel(image_dilated)

    # Update markers based on dilated image
    markers = np.zeros_like(image_dilated, dtype=int)
    markers[image_dilated < 0.5] = 2
    markers[image_dilated > 0.6] = 1

    # Perform watershed region segmentation on grayscale
    segmentation_result = segmentation.watershed(elevation_map, markers)
    segmentation_filled = nd.binary_fill_holes(segmentation_result - 1)

    # Step 3: Create a 20-pixel padded contour mask
    padded_segmentation = dilation(segmentation_filled, disk(20))

    # Step 4: Initialize the output image in color and apply the contour condition
    output_image = np.copy(image)  # Start with the original color image

    threshold = 0.4
    percentage_threshold = 0.7

    # Label the padded contours for each unique region
    labeled_contours, num_contours = measure.label(padded_segmentation, return_num=True)

    # Iterate over each contour
    for contour_label in range(1, num_contours + 1):
        # Mask for the current contour
        contour_mask = (labeled_contours == contour_label)

        # Calculate intensities within the contour on grayscale image
        contour_intensities = grayscale_image[contour_mask]
        below_threshold = np.sum(contour_intensities < threshold)
        percentage_below_threshold = below_threshold / len(contour_intensities) if len(contour_intensities) > 0 else 0

        # Apply condition: If 70% of pixels are below threshold, set outside contour to white
        if percentage_below_threshold >= percentage_threshold:
            output_image[~contour_mask] = [255, 255, 255]  # Set outside to white for color

    # Display the final output image
    plt.imshow(output_image)
    plt.title("Final Output Image with Condition Applied")
    plt.axis('off')
    plt.show()

    # Save the image if save_path is specified
    if save_path:
        imsave(save_path, output_image)

    return output_image


# ------- Loop through all PNG files in the directory --------
img_directory = "static/inputs/"
i=1
for imgPath in glob.glob(os.path.join(img_directory, "*.jpg")):
    print(f"Processing ({i}/328){((i/328)*100):.2f}%: {imgPath}")

    # ------- Call The Cleaning & Processing Function --------
    cleaned_processed_image = edge_detect_and_noise_remover_colored(imread(imgPath))

    # Construct output file path
    original_filename = os.path.basename(imgPath)  # Get the original image name
    output_path = os.path.join(output_directory, original_filename)
    imsave(output_path, cleaned_processed_image)

    i+=1


