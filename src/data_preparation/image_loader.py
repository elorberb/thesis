import os
import cv2


def extract_file_name(file_path):
    """
    Extracts the file name from a file path.
    """
    # Split the file path into a list of strings
    parts = file_path.split("/")
    # Get the last element in the list (the file name)
    file_name = parts[-1]
    # Split the file name into a list of strings
    parts = file_name.split(".")
    # Get the first element in the list (the file name without the extension)
    file_name = parts[0]

    return file_name


def read_images_and_names(dir_path, func=None, verbose=False):
    """
    Read all images and their corresponding names in a directory and return them as a list of tuples.
    Each tuple in the list contains a NumPy array representing the image and a string representing the name of the image.

    Parameters:
    - dir_path (str): Path to the directory containing images.
    - func (function, optional): A function to apply to each image after reading.
    - verbose (bool): Flag to print information about the ongoing operations. Defaults to False.

    Returns:
    - list of tuples: A list containing tuples of images and their corresponding names.
    """
    images_and_names = []

    for filename in os.listdir(dir_path):
        # Check if file is an image
        if filename.endswith((".png", ".jpg", ".PNG", ".JPG", ".jpeg")):
            # Read image and store as NumPy array
            file_path = os.path.join(dir_path, filename)
            image = cv2.imread(file_path)

            # If a function is provided, apply it to the image
            if func is not None:
                image = func(image)

            image_name = extract_file_name(file_path)
            images_and_names.append((image, image_name))

            # If verbose is True, print the image being processed
            if verbose:
                print(f"Processed image: {filename}")

    return images_and_names
