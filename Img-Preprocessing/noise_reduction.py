# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from skimage.filters import threshold_otsu
from skimage.util import view_as_windows

# Functions for image preprocessing
def binary_threshold(image: np.ndarray, threshold=127) -> np.ndarray:
    """
    Apply binary thresholding to the image.
    Args:
        image: Input image in grayscale.
        threshold: Threshold value for binarization.
    
    Returns:
        Binary image after applying thresholding.
    """
    # Ensure image is uint8 in [0, 255]
    if image.dtype != np.uint8:
        img = (image * 255).round().astype(np.uint8)
    else:
        img = image
    _, binary_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def adaptive_binary_threshold(image: np.ndarray, block_size=15, C=2) -> np.ndarray:
    """
    Apply adaptive binary thresholding to the image.
    Args:
        image: Input image in grayscale.
        block_size: Size of a pixel neighborhood that is used to calculate a threshold value.
        C: Constant subtracted from the mean or weighted mean.
    Returns:
        Binary image after applying adaptive thresholding.
    """
    # Ensure image is uint8 in [0, 255]
    if image.dtype != np.uint8:
        img = (image * 255).round().astype(np.uint8)
    else:
        img = image
    binary_image = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, C
    )
    return binary_image

def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from the image using morphological operations.
    Args:
        image: Input binary image.
    Returns:
        Denoised image.
    """
    kernel = np.ones((3, 3), np.uint8)
    denoised_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return denoised_image

# Contrast enhancement function using CLAHE
def enhance_contrast(image: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    Args:
        image: Input image in grayscale.
        clip_limit: Contrast limit for CLAHE.
        tile_grid_size: Size of the grid for histogram equalization.
    Returns:
        Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    return clahe_image

# Canny edge detection function
def preprocess_and_canny(image: np.ndarray, 
                         blur_ksize=5, 
                         morph_ksize=3, 
                         canny_thresh1=30, 
                         canny_thresh2=80) -> np.ndarray:
    """
    Preprocess image (Gaussian blur + morphological opening) and apply Canny edge detection.
    Args:
        image: Input grayscale image (float32 [0,1] or uint8 [0,255]).
        blur_ksize: Kernel size for Gaussian blur.
        morph_ksize: Kernel size for morphological opening.
        canny_thresh1: Lower threshold for Canny.
        canny_thresh2: Upper threshold for Canny.
    Returns:
        Edge-detected binary image.
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        img = (image * 255).round().astype(np.uint8)
    else:
        img = image
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    # Morphological opening
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    # Canny edge detection
    edges = cv2.Canny(opened, canny_thresh1, canny_thresh2)
    return edges

# refined edge detection function
def refined_canny_edge_detection(image: np.ndarray,
                                   blur_ksize=7,
                                   median_ksize=5,
                                   canny_thresh1=50,
                                   canny_thresh2=120,
                                   morph_ksize=(3, 15)) -> np.ndarray:
    """
    Refined edge detection for OCT retinal layers.
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        img = (image * 255).round().astype(np.uint8)
    else:
        img = image
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    # Median filter
    median = cv2.medianBlur(blurred, median_ksize)
    # Canny edge detection
    edges = cv2.Canny(median, canny_thresh1, canny_thresh2)
    # Morphological closing (horizontal)
    kernel = np.ones(morph_ksize, np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

# Sobel edge detection in y-direction
def sobel_y_edge_detection(image: np.ndarray, ksize=3) -> np.ndarray:
    """
    Apply Sobel edge detection in the y-direction (vertical gradient).
    Args:
        image: Input image in grayscale.
        ksize: Kernel size for the Sobel operator (must be odd).
    Returns:
        Edge-detected image (Sobel y).
    """
    # Ensure image is uint8 in [0, 255]
    if image.dtype != np.uint8:
        img = (image * 255).round().astype(np.uint8)
    else:
        img = image
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_y = np.absolute(sobel_y)
    sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))  # Normalize to [0,255]
    return sobel_y
# Load numpy image from .h5 file

#data_dir = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control_normalized_corrected.h5'
data_dir = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control_normalized.h5'
with h5py.File(data_dir, 'r') as f:
    images = f['images'][:]
    print(f.keys())
    print("Image shape:", images.shape)

# Convert from float32 [0, 1] to uint8 [0, 255]
img_float32 = images[14]  # already in float32, [0, 1]
img_uint8 = (img_float32 * 255).round().astype(np.uint8)


# Convert from uint8 [0, 255] back to float32 [0, 1]
# img_float32_recovered = img_uint8.astype(np.float32) / 255.0


# Apply noise reduction to the image
denoised_image = remove_noise(images[14])

#  Apply binary thresholding to the denoised image
binary_image = binary_threshold(denoised_image)


# Apply adaptive binary thresholding to the original image
adaptive_binary_image = adaptive_binary_threshold(images[14])


# Apply contrast enhancement to the original image
img = images[14]
if img.dtype != np.uint8:
    img = (img * 255).round().astype(np.uint8)
clahe_image = enhance_contrast(img)

# Apply Canny edge detection to the original image
canny_edges = preprocess_and_canny(clahe_image)

# Refined edge detection for retinal layers
edges = refined_canny_edge_detection(images[14])

# Sobel edge detection in the y-direction
sobel_y_img = sobel_y_edge_detection(clahe_image)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Original image
plt.imshow(images[14], cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "original_image.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# Denoised image
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "denoised_image.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# Binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "binary_image.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# Adaptive binary image
plt.imshow(adaptive_binary_image, cmap='gray')
plt.title('Adaptive Binary Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "adaptive_binary_image.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# CLAHE image
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Image')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "clahe_image.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# Canny edges
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "canny_edges.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# Refined retinal layer edges
plt.imshow(edges, cmap='gray')
plt.title('Refined Retinal Layer Edges')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "refined_retinal_layer_edges.png"), bbox_inches='tight', pad_inches=0)
plt.show()

# Sobel Y edges
plt.imshow(sobel_y_img, cmap='gray')
plt.title('Sobel Y Edges')
plt.axis('off')
plt.savefig(os.path.join(output_dir, "sobel_y_edges.png"), bbox_inches='tight', pad_inches=0)
plt.show()