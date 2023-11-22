import pandas as pd
import cv2
import matplotlib.pyplot as plt
import config

def show_patch_on_raw_image(patch_filename):
    """
    Show the raw image with an annotated bounding box around the specified patch.

    Parameters:
    patch_filename (str): Filename of the patch (e.g., 'IMG_2242_p0.png').
    """
    # Extract the image number from the patch filename
    image_number = patch_filename.split('_p')[0]

    # Read metadata CSV to find the coordinates
    metadata_df = pd.read_csv(config.CANNABIS_PATCHES_METADATA_FILE)
    patch_metadata = metadata_df[metadata_df['patch_name'] == patch_filename].iloc[0]
    x, y = patch_metadata['x'], patch_metadata['y']

    # Find the week and zoom type for the image
    week, zoom_type = config.find_image_details(image_number)
    if not week or not zoom_type:
        print("Image details not found.")
        return

    # Get the path to the raw image
    raw_image_path = config.get_raw_image_path(week, zoom_type) / f"{image_number}.JPG"

    # Read the raw image
    raw_image = cv2.imread(str(raw_image_path))
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)  # Convert color to RGB

    # Calculate the bounding box
    bbox = [x, y, x + config.CANNABIS_PATCH_SIZE, y + config.CANNABIS_PATCH_SIZE]

    # Draw the bounding box on the image
    cv2.rectangle(raw_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    # Display the image
    plt.imshow(raw_image)
    plt.title(f"Bounding Box on {image_number}")
    plt.show()
