# Import all functions and classes from the preprocess.image_preprocess module
from preprocess.image_preprocess import *

if __name__ == '__main__':
    # Define the path to the source images
    images_path = "/sise/home/etaylor/images/week5-18.5.23/3xzoom_regular"

    # Define the directory where the preprocessed patches will be saved
    saving_dir = "/sise/home/etaylor/images/trichomes_patches/week5_3xzoom_regular"

    print("Starting reading Images...")
    # Read images and their names from the specified directory
    trichomes_images = read_images_and_names(dir_path=images_path, verbose=True)

    # Preprocess the images: This involves cutting each image into patches 
    # and then filtering and saving the patches to the specified directory.
    # The verbose mode provides detailed logging of each step in the preprocessing.
    preprocess_patches(saving_dir_path=saving_dir, trichomes_images=trichomes_images, verbose=True)
