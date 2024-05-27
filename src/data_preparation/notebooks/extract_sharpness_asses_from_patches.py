from src.data_preparation.sharpness_assessment import calculate_sharpness
import os
import json
import cv2


def calculate_and_save_sharpness(images_dir, save_path):
    """
    Calculate sharpness for all image patches in a directory and save the results in a JSON file.
    
    Args:
        images_dir (str): Directory containing image patches.
        save_path (str): Path to save the JSON file with sharpness scores.
    """
    sharpness_dict = {}

    # Traverse the directory and process each image
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith('.png'):  # Process only PNG files
                print(f"Processing image: {file}")
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                sharpness_score = calculate_sharpness(image)
                sharpness_dict[file] = sharpness_score

    # Save the sharpness scores to a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(sharpness_dict, json_file, indent=4)

# Define the paths
images_dir = "/home/etaylor/images/processed_images/cannabis_patches"
save_path = "/home/etaylor/code_projects/thesis/data/metadata/sharpness_per_patch_scores.json"

# Calculate and save sharpness scores
calculate_and_save_sharpness(images_dir, save_path)