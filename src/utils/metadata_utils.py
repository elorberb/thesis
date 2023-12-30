import csv
import os
import config
from datetime import datetime


def annotation_tracking(csv_file, image_number, annotator):
    # Get the current date and time of the model training process
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    # Check if the CSV file exists to determine if the header is needed
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # If the file is new, write the header
        if write_header:
            writer.writerow(['Image Number', 'Annotator', 'Time'])

        # Write the annotation record
        writer.writerow([image_number, annotator, current_time])
        
        
if __name__ == "__main__":
    # Test the annotation_tracking function
    dorin = "dorins"
    nir = "Nirmalka"
    etay = "etaylor"
    print("Updating annotations_tracking.csv file...")
    annotation_tracking(config.ANNOTATIONS_TRACKING_METADATA_FILE, image_number="0016", annotator=etay)
    annotation_tracking(config.ANNOTATIONS_TRACKING_METADATA_FILE, image_number="0017", annotator=etay)
    print("annotations_tracking.csv file updated successfully!")






