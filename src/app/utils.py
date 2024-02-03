import base64
import pandas as pd
import constants as const
from google.cloud import storage


def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# Experiment utils
def load_random_image(reviewed_images):
    client = storage.Client()
    bucket = client.bucket(const.BUCKET_NAME)

    # Prefix to filter images if necessary, e.g., 'images/' to list all images
    prefix = 'images/'
    blobs = client.list_blobs(bucket, prefix=prefix)

    # Create a list of image paths from blobs that are not in reviewed_images
    available_images = [blob.name for blob in blobs if
                        blob.name.split('/')[-1].replace('.JPG', '') not in reviewed_images]

    if available_images:
        # Choose a random image from available ones
        random_image_path = random.choice(available_images)
        image_number = random_image_path.split('/')[-1].replace('.JPG', '')


        # Return the full URL to access the image directly
        image_url = f"gs://{const.BUCKET_NAME}/{random_image_path}"
        return image_url, image_number
    else:
        return None, None