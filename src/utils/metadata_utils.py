import csv
import os
import config

def annotation_tracking(csv_file, image_number, annotator):
    # Check if the CSV file exists to determine if the header is needed
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # If the file is new, write the header
        if write_header:
            writer.writerow(['Image Number', 'Annotator'])

        # Write the annotation record
        writer.writerow([image_number, annotator])
        
        
if __name__ == "__main__":
    # Test the annotation_tracking function
    dorin = "dorins"
    nir = "Nirmalka"
    annotation_tracking(config.ANNOTATIONS_TRACKING_METADATA_FILE, image_number=2129, annotator=nir)


