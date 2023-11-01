from src.data_preparation.image_loader import read_images_and_names
from src.data_preparation.patch_cutter import preprocess_patches


def preprocessing_pipeline(images_path, saving_dir, verbose=False):
    """
    Preprocess images from a given directory and save the processed patches.

    Parameters:
    - images_path (str): Path to the directory containing the images to preprocess.
    - saving_dir (str): Directory where the preprocessed patches will be saved.
    - verbose (bool, optional): If True, print progress messages. Defaults to False.
    """
    
    trichomes_images = read_images_and_names(dir_path=images_path, verbose=verbose)
    preprocess_patches(saving_dir_path=saving_dir, trichomes_images=trichomes_images, verbose=verbose)


# if __name__ == '__main__':
#     images_path = "/sise/home/etaylor/images/week9-15.6.23/3xzoom_regular"
#     saving_dir = "/sise/home/etaylor/images/cannabis_patches/week9_3xzoom_regular_v2"
#     verbose = True
#     preprocessing_pipeline(images_path=images_path, saving_dir=saving_dir, verbose=verbose)