from pathlib import Path
import os

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

# --- METADATA CONFIGURATION ---
CANNABIS_PATCHES_METADATA_FILE = 'metadata/cannabis_patches_metadata.csv'
ANNOTATIONS_TRACKING_METADATA_FILE = 'metadata/annotations_tracking.csv'
CANNABIS_PATCH_SIZE = 512

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


if __name__ == "__main__":
    # Example usage:
    raw_image_path = get_raw_image_path('week9', '1xr')
    print(f"Path for raw images for week 2, 1x regular: {raw_image_path}")

    processed_cannabis_image_path = get_processed_cannabis_image_path('week9', '3xr')
    print(f"Path for processed cannabis images for week 2, 1x regular: {processed_cannabis_image_path}")

    processed_trichome_image_path = get_processed_trichome_image_path('week9', '3xr')
    print(f"Path for processed trichome images for week 2, 1x regular: {processed_trichome_image_path}")
    
    # Example usage for find_image_details
    image_number = "IMG_9969"
    week, zoom_type = find_image_details(image_number)
    if week and zoom_type:
        print(f"Week: {week}, Zoom Type: {zoom_type}")
    else:
        print("Image not found.")
