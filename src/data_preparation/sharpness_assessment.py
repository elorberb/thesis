import cv2
import numpy as np


# Gradient-based sharpness
def calculate_gradient_based_sharpness(image):
    # Convert image to grayscale if it is not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Calculate gradients along the x and y axis
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    gnorm = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate the average sharpness
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
