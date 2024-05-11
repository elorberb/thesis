from pathlib import Path
import os
from datetime import datetime
import csv

# --- IMAGES CONFIGURATION ---

# Define root path of the project
ROOT_IMAGES_DIR = Path("/sise/home/etaylor/images/")

# Define different image paths
RAW_IMAGE_DIR = ROOT_IMAGES_DIR  / 'raw_images'
PROCESSED_IMAGE_DIR = ROOT_IMAGES_DIR / 'processed_images'
PROCESSED_CANNABIS_PATCHES_DIR = PROCESSED_IMAGE_DIR / 'cannabis_patches'
PROCESSED_TRICHOME_PATCHES_DIR = PROCESSED_IMAGE_DIR / 'trichome_patches'

# csv of good quality images
GOOD_QUALITY_IMAGES_CSV = Path('/home/etaylor/code_projects/thesis/metadata/good_quality_images.csv')

# Define specific paths for weeks and zoom types
WEEKS_DIR = {
    'week2': 'week2_27_04_2023',
    'week3': 'week3_03_05_2023',
    'week4': 'week4_09_05_2023',
    'week5': 'week5_18_05_2023',
    'week6': 'week6_22_05_2023',
    'week7': 'week7_01_06_2023',
    'week8': 'week8_07_06_2023',
    'week9': 'week9_15_06_2023',
    # Add other weeks as necessary
}

ZOOM_TYPES_DIR = {
    '1xr': '1x_regular',
    '1xfs': '1x_focus_stacking',
    '3xr': '3x_regular',
    '3xfs': '3x_focus_stacking',
}

ANNOTATIONS_CLASS_MAPPINGS = {
            0: 'trichome',
            1: 'clear',
            2: 'cloudy',
            3: 'amber'
        }

# --- METADATA CONFIGURATION ---
CANNABIS_PATCHES_METADATA_FILE = 'data/metadata/cannabis_patches_metadata.csv'
ANNOTATIONS_TRACKING_METADATA_FILE = 'data/metadata/annotations_tracking.csv'
CANNABIS_PATCH_SIZE = 512

# --- DATABASE CONFIGURATION ---
EXPERIMENT_DATABASE_FILE = 'src/app/experiment_db.db'

# --- SEGMENTS FOLDER PATH ---
SEGMENTS_FOLDER = '/home/etaylor/code_projects/thesis/segments'

DATETIME_STR_FORMAT = '%d-%m-%Y_%H-%M-%S'
def get_datetime_str():
    """Return the current date and time as a formatted string."""
    return datetime.now().strftime(DATETIME_STR_FORMAT)

# ultralytics settings configuration
ULTRALYTICS_RUNS_DIR = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/runs"
ULTRALYTICS_WEIGHTS_DIR = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/weights"
ULTRALYTICS_DATASETS_DIR = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/datasets"

# Example of using a function to get a specific path
def get_raw_image_path(week, zoom_type):
    """Return the path for the raw images of a given week and zoom type."""
    week_dir = WEEKS_DIR.get(week, week)  # Get value if week is a key, else use week as is
    zoom_type_dir = ZOOM_TYPES_DIR.get(zoom_type, zoom_type)  # Get value if zoom_type is a key, else use zoom_type as is
    return RAW_IMAGE_DIR / week_dir / zoom_type_dir


def get_processed_cannabis_image_path(week, zoom_type):
    """Return the path for the processed cannabis images of a given week and zoom type."""
    week_dir = WEEKS_DIR.get(week, week)
    zoom_type_dir = ZOOM_TYPES_DIR.get(zoom_type, zoom_type)
    return PROCESSED_CANNABIS_PATCHES_DIR / week_dir / zoom_type_dir


def get_processed_trichome_image_path(week, zoom_type):
    """Return the path for the processed trichome images of a given week and zoom type."""
    week_dir = WEEKS_DIR.get(week, week)
    zoom_type_dir = ZOOM_TYPES_DIR.get(zoom_type, zoom_type)
    return PROCESSED_TRICHOME_PATCHES_DIR / week_dir / zoom_type_dir


def find_image_details(image_number, base_path=RAW_IMAGE_DIR):
    """
    Find the week and zoom type for a given image number.

    Parameters:
    image_number (str): The number of the image (e.g., 'IMG_2242').
    base_path (str): The base path where images are stored.

    Returns:
    tuple: (week, zoom_type) if found, otherwise (None, None).
    """
    for week in os.listdir(base_path):
        week_path = os.path.join(base_path, week)
        if os.path.isdir(week_path):
            for zoom_type in os.listdir(week_path):
                zoom_path = os.path.join(week_path, zoom_type)
                if os.path.isdir(zoom_path):
                    image_path = os.path.join(zoom_path, f"{image_number}.JPG")
                    if os.path.exists(image_path):
                        return week, zoom_type
    return None, None

def get_image_path(image_name, base_path=RAW_IMAGE_DIR, processed_type=None):
    """
    Return the path for an image given its name and optional processing type.

    Parameters:
    image_name (str): The name of the image (e.g., 'IMG_2242').
    base_path (str): The base path where images are stored. Defaults to RAW_IMAGE_DIR.
    processed_type (str): The type of processed image ('cannabis' or 'trichome') if applicable.

    Returns:
    str: The path to the image, if found, otherwise None.
    """
    # Find the week and zoom type for the image
    week, zoom_type = find_image_details(image_name, base_path)
    
    if week is not None and zoom_type is not None:
        week_dir = WEEKS_DIR.get(week, week)
        zoom_type_dir = ZOOM_TYPES_DIR.get(zoom_type, zoom_type)
        
        if processed_type == 'cannabis':
            return os.path.join(PROCESSED_CANNABIS_PATCHES_DIR, week_dir, zoom_type_dir, f"{image_name}.JPG")
        elif processed_type == 'trichome':
            return os.path.join(PROCESSED_TRICHOME_PATCHES_DIR, week_dir, zoom_type_dir, f"{image_name}.JPG")
        else:
            # For raw images or if no processed_type is specified
            return os.path.join(base_path, week_dir, zoom_type_dir, f"{image_name}.JPG")
    
    return None

# Metadata handling functions
def add_annotation_tracking_entry(image_number, annotator, csv_path=ANNOTATIONS_TRACKING_METADATA_FILE):
    # Create a timestamp for the entry
    time_stamp = datetime.now().strftime(DATETIME_STR_FORMAT)
    # Set the done flag to False
    done_flag = 'False'
    # New entry to add
    new_entry = [image_number, annotator, time_stamp, done_flag]
    
    # Open the CSV file and append the new entry
    with open(csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(new_entry)
        
        
def update_annotation_status(image_number, csv_path=ANNOTATIONS_TRACKING_METADATA_FILE):
    # Create a timestamp for the update
    time_stamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    # Temporary list to hold updated rows
    updated_rows = []

    # Read the CSV, update the necessary row, and store the updates
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0] == image_number:
                # Update the 'done' flag to True and the time
                row[2] = time_stamp
                row[3] = 'True'
            updated_rows.append(row)

    # Write the updated content back to the CSV
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(updated_rows)


if __name__ == "__main__":
    # # Example usage:
    # raw_image_path = get_raw_image_path('week9', '1xr')
    # print(f"Path for raw images for week 2, 1x regular: {raw_image_path}")

    # processed_cannabis_image_path = get_processed_cannabis_image_path('week9', '3xr')
    # print(f"Path for processed cannabis images for week 2, 1x regular: {processed_cannabis_image_path}")

    # processed_trichome_image_path = get_processed_trichome_image_path('week9', '3xr')
    # print(f"Path for processed trichome images for week 2, 1x regular: {processed_trichome_image_path}")
    
    # # Example usage for find_image_details
    # image_number = "IMG_9969"
    # week, zoom_type = find_image_details(image_number)
    # if week and zoom_type:
    #     print(f"Week: {week}, Zoom Type: {zoom_type}")
    # else:
    #     print("Image not found.")
        
    # # Example of using get_image_path
    # image_path = get_image_path(image_number)
    # print(f"Path for image {image_number}: {image_path}")
    
    # # METADATA annotation handling function Usage example:
    # # To add a new image for annotation
    # add_annotation_tracking_entry('IMG_test', 'etaylor')

    # To update an existing image as annotated
    # print("Updating annotation status for images...")
    update_annotation_status('IMG_2161')
    update_annotation_status('IMG_1145')
    update_annotation_status('IMG_0577')
    update_annotation_status('IMG_1193')
    update_annotation_status('IMG_0642')
    # print("Annotation status updated successfully!")
