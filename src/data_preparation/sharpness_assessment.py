import cv2
import numpy as np


# Gradient-based sharpness
# TODO - check if there is a gradient calculation with cv2 - using kernel.
def calculate_image_sharpness(image):
    array = np.asarray(image, dtype=np.int32)
    gradients = np.gradient(array)
    gnorm_squares = np.sum([g**2 for g in gradients], axis=0)
    gnorm = np.sqrt(gnorm_squares)
    sharpness = np.average(gnorm)
    return sharpness


# Laplacian-based sharpness
def laplacian_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    return sharpness


# Edge-based sharpness
def edge_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 100, 200)
    sharpness = np.sum(edges) / float(edges.size)
    return sharpness


# Tenengrad-based sharpness
def tenengrad_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = np.mean(gx**2 + gy**2)
    return sharpness


def fft_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    f_transform = np.fft.fft2(gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    sharpness = np.mean(magnitude_spectrum)
    return sharpness


def contrast_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    contrast = np.std(gray)
    return contrast


def calculate_sharpness(image):
    """
    Currently sharpness function for the pipeline.
    """
    return edge_sharpness(image)


# ------- Old sharpness functions -------


def calculate_sharpness_old(image):
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the sharpness
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness


def is_monochromatic_old(image, tolerance=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    if h_std > tolerance or s_std > tolerance or v_std > tolerance:
        return False
    else:
        return True


def is_blurry_old(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold
