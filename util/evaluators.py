import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import scipy.stats as stats

# def calculate_entropy(image):
#     """ Calculate the entropy of an image. """
#     hist = cv2.calcHist([image], [0], None, [256], [0,256])
#     hist_normalize = hist.ravel()/hist.sum()
#     entropy = -np.sum(hist_normalize*np.log2(hist_normalize + np.finfo(float).eps))  # Adding epsilon to avoid log2(0)
#     return entropy

def calculate_standard_deviation(image):
    """Calculate the standard deviation of an image."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Convert back to 0-255 scale if the image is in [0, 1]
        image = (image * 255).astype(np.uint8)
    return np.std(image)


def mutual_information(hgram):
    """ Calculate the mutual information based on a joint histogram. """
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Non-zero selections
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def calculate_ssim(image1, image2):
    """ Calculate the Structural Similarity Index between two images. """
    return ssim(image1, image2, multichannel=True)

def calculate_mi(image1, image2):
    """ Calculate Mutual Information between two images. """
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=256)
    return mutual_information(hist_2d)

def reload_and_preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Normalize and convert to float if necessary
    image = image.astype('float32') / 255.0
    return image

def calculate_entropy(image):
    """ Calculate the entropy of an image. """
    # Ensure the image is in a suitable format for calcHist
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_normalize = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_normalize * np.log2(hist_normalize + np.finfo(float).eps))  # Adding epsilon to avoid log2(0)
    return entropy


# # Load images
# imageA = cv2.imread('path_to_image_A.jpg', cv2.IMREAD_GRAYSCALE)  # Adjust as needed
# imageB = cv2.imread('path_to_image_B.jpg', cv2.IMREAD_GRAYSCALE)  # Adjust as needed
# imageF = cv2.imread('path_to_image_F.jpg', cv2.IMREAD_GRAYSCALE)  # Adjust as needed

# # Calculate metrics
# entropyA = calculate_entropy(imageA)
# std_devA = calculate_standard_deviation(imageA)
# ssimAF = calculate_ssim(imageA, imageF)
# miAF = calculate_mi(imageA, imageF)

# # Print results
# print("Entropy of Image A:", entropyA)
# print("Standard Deviation of Image A:", std_devA)
# print("SSIM between A and F:", ssimAF)
# print("Mutual Information between A and F:", miAF)

# # Add similar calculations for image B and F, and other metrics as needed
