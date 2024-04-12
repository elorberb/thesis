import numpy as np
import os
import shutil
from segments.utils import export_dataset
from src.annotation_handling.segmentsai_handler import SegmentsAIHandler

# YOLO utils
import random
from pathlib import Path
from ultralytics.data.converter import convert_coco
import config

SEGMENTS_HANDLER = SegmentsAIHandler()

def convert_coco_to_segments_format(image, outputs):
    segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
    annotations = []
    counter = 1
    instances = outputs['instances']
    for i in range(len(instances.pred_classes)):
        category_id = int(instances.pred_classes[i])
        instance_id = counter
        mask = instances.pred_masks[i].cpu()
        segmentation_bitmap[mask] = instance_id
        annotations.append({'id': instance_id, 'category_id': category_id})
        counter += 1
    return segmentation_bitmap, annotations


def convert_segments_to_coco_format(dataset_name, release_version, export_format="coco-instance", output_dir="."):
    # get the dataset instance
    dataset = SEGMENTS_HANDLER.get_dataset_instance(dataset_name, version=release_version)
        
    # export the dataset - format is coco instance segmentation
    export_json_path, saved_images_path = export_dataset(dataset, export_format=export_format, export_folder=output_dir)

    # Create the annotations folder one level up from saved_images_path
    annotations_folder = os.path.join(os.path.dirname(saved_images_path), "annotations")
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

    # Move the export_json_path file to the annotations folder
    new_export_json_path = os.path.join(annotations_folder, os.path.basename(export_json_path))
    shutil.move(export_json_path, new_export_json_path)

    return dataset, new_export_json_path, saved_images_path


# ----- YOLO annotations functions -----

def split_data(image_dir, label_dir, train_percentage, output_base_dir):
    """
    Splits image and label data into training and validation sets, and organizes them into specified directories.

    Parameters:
    image_dir (str): The directory where the input images are stored.
    label_dir (str): The directory where the corresponding labels for the images are stored.
    train_percentage (float): The percentage of the dataset to be used as the training set (between 0 and 1).
    output_base_dir (str): The base directory where the train/val split directories will be created.

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

    Parameters:
    dataset_path (str): The base path where the organized dataset is located.
    yaml_path (str): The path (including filename) where the YAML file will be saved.
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

def convert_coco_to_yolo_format(annotations_folder_name, dataset_version, saving_yaml_path, train_percentage=0.8):
    """
    Converts COCO annotations to YOLO format, splits the data into training and validation sets,
    and creates a YAML file for training configuration.

    :param segments_base_folder: Base folder path for segments.
    :param annotations_folder_name: Folder name containing COCO annotations.
    :param images_folder_name: Folder name containing images.
    :param yaml_path: Path to save the YAML configuration file.
    :param train_percentage: Percentage of data to use for training (default is 0.8).
    """
    # Construct paths
    annotations_dir = f"{config.SEGMENTS_FOLDER}/{annotations_folder_name}/annotations"
    output_dir = f"{annotations_dir}/yolo"
    image_dir = f"{config.SEGMENTS_FOLDER}/{annotations_folder_name}/{dataset_version}"
    label_dir = f"{output_dir}/labels/export_coco-instance_{annotations_folder_name}_{dataset_version}"
    organized_dataset_path = f"{output_dir}_split"

    # Convert annotations from COCO to YOLO format
    convert_coco(
        labels_dir=annotations_dir,
        save_dir=output_dir,
        use_segments=False,
    )

    # Split data into training and validation sets
    split_data(
        image_dir=image_dir,
        label_dir=label_dir,
        train_percentage=train_percentage,
        output_base_dir=organized_dataset_path
    )
    
    # Create YAML file for YOLO training
    saving_yaml_file_path = os.path.join(saving_yaml_path, f"{annotations_folder_name}_{dataset_version}_data.yaml")
    create_yaml(organized_dataset_path, saving_yaml_file_path)
    
    # remove old yolo folder because we use only yolo_split folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"The folder {output_dir} has been deleted.")
    else:
        print(f"The folder {output_dir} does not exist.")
        
    return saving_yaml_file_path