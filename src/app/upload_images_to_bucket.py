import csv
from google.cloud import storage
import config  # Make sure this is correctly pointing to your config file/module

def upload_files(bucket_name):
    """Uploads files listed in the CSV to the specified GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Assume config.GOOD_QUALITY_IMAGES_CSV is the path to your CSV file
    csv_file_path = config.GOOD_QUALITY_IMAGES_CSV

    with open(csv_file_path, newline='') as csvfile:
        filereader = csv.DictReader(csvfile)  # Use DictReader to easily access columns by name
        for row in filereader:
            week_number = row['week_number']
            image_number = row['image_number']
            
            # Retrieve week and zoom_type based on image_number (assuming this function exists and returns correct values)
            week, zoom_type = config.find_image_details(image_number)
            
            # Assuming get_raw_image_path returns a Path object and concatenating the filename to it
            if week and zoom_type:
                image_path = config.get_raw_image_path(week, zoom_type) / f"{image_number}.JPG"
                blob = bucket.blob(f"images/{image_number}.JPG")  # Organizing files by week_number in the bucket
                blob.upload_from_filename(str(image_path))  # Convert Path object to string if necessary
                print(f"Uploaded {image_number} to {bucket_name}/{week_number}/")
            else:
                print(f"Details not found for image number: {image_number}")

# Example usage
bucket_name = 'trichome_classification_study_storage'
upload_files(bucket_name)
