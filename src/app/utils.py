import base64
import pandas as pd
import random

def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def load_random_image(reviewed_images):
    # Define the path to your CSV file
    csv_file_path = 'resources/good_quality_images.csv'

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Filter out any images that have already been reviewed
    available_images = df[~df['image_number'].isin(reviewed_images)]

    if not available_images.empty:
        # Randomly select a row (an image) from the available images
        random_row = available_images.sample(n=1).iloc[0]
        image_number = random_row['image_number']

        # Construct the URL to the image based on its number and week
        # Assuming a URL pattern like: https://storage.googleapis.com/[BUCKET_NAME]/images/[week_number]/[image_number].JPG
        bucket_name = 'trichome_classification_study_storage'  # Replace with your actual GCS bucket name
        image_url = f"https://storage.googleapis.com/{bucket_name}/images/{image_number}.JPG"

        return image_url, image_number
    else:
        return None, None
