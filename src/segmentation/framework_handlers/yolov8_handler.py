import os
import shutil
import random
from pathlib import Path


def split_data(image_dir, label_dir, train_percentage, output_base_dir):
    """
    Splits image and label data into training and validation sets, and organizes them into specified directories.

    This function creates training and validation directories for both images and labels within a specified base directory.
    It then divides the images (and corresponding labels, if present) into these directories based on the specified training
    percentage. The division is done randomly.

    Parameters:
    image_dir (str): The directory where the input images are stored.
    label_dir (str): The directory where the corresponding labels for the images are stored.
    train_percentage (float): The percentage of the dataset to be used as the training set (between 0 and 1).
    output_base_dir (str): The base directory where the train/val split directories will be created.

    Returns:
    None: This function does not return anything but creates training and validation sets in the specified directories.

    Note:
    - The function assumes that each image has a corresponding label file with the same name but a '.txt' extension.
    - If a label file is missing for an image, it is ignored and not included in the split.
    - The function uses symbolic links for images and copies the label files.
    """
    train_img_dir = os.path.join(output_base_dir, 'images/train')
    val_img_dir = os.path.join(output_base_dir, 'images/val')
    train_label_dir = os.path.join(output_base_dir, 'labels/train')
    val_label_dir = os.path.join(output_base_dir, 'labels/val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # List all images
    all_images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Shuffle and split images into train and val sets
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_percentage)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # Function to create symbolic links
    def create_symlinks(images, source_dir, target_img_dir, target_label_dir):
        for img_name in images:
            src_img_path = os.path.join(source_dir, img_name)
            dst_img_path = os.path.join(target_img_dir, img_name)
            os.symlink(src_img_path, dst_img_path)

            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label_path = os.path.join(label_dir, label_name)
            dst_label_path = os.path.join(target_label_dir, label_name)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)

    # Create symlinks and copy annotations for train and val sets
    create_symlinks(train_images, image_dir, train_img_dir, train_label_dir)
    create_symlinks(val_images, image_dir, val_img_dir, val_label_dir)

def create_yaml(dataset_path, yaml_path):
    """
    Creates a YAML file with dataset paths and class names.

    This function writes a YAML file specifying the paths for training and validation data, 
    as well as the names of the classes for a machine learning project.

    Parameters:
    dataset_path (str): The base path where the organized dataset is located.
    yaml_path (str): The path (including filename) where the YAML file will be saved.

    Returns:
    None: The function writes to a file and does not return any value.
    """

    yaml_content = f'''
    path: {dataset_path}
    train: images/train
    val: images/val

    names:
      0: trichome
      1: clear
      2: cloudy
      3: amber
    '''

    with Path(yaml_path).open('w') as f:
        f.write(yaml_content)