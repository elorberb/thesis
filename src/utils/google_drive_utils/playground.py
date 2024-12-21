import os
import rawpy
from PIL import Image


def convert_dng_to_jpg_recursive(input_folder):
    """
    Recursively converts DNG files to JPG in a given folder and its subfolders.

    Args:
        input_folder (str): Path to the root folder.
    """
    print(f"Starting conversion process in folder: {input_folder}")

    # Walk through the directory
    for root, dirs, files in os.walk(input_folder):
        print(f"Entering directory: {root}")
        for file in files:
            # Check if the file is a DNG file
            if file.lower().endswith(".dng"):
                dng_path = os.path.join(root, file)

                # Define output path (same folder, same filename but JPG extension)
                jpg_path = os.path.splitext(dng_path)[0] + ".JPG"

                try:
                    print(f"Converting: {dng_path} -> {jpg_path}")

                    # Read and process DNG image
                    with rawpy.imread(dng_path) as raw:
                        print(f"Reading DNG file: {dng_path}")
                        rgb = raw.postprocess()
                        print(f"Processing DNG file: {dng_path}")

                    # Convert to JPG and save
                    img = Image.fromarray(rgb)
                    print(f"Saving JPG file: {jpg_path}")
                    img.save(jpg_path, "JPEG")

                    print(f"Successfully converted: {jpg_path}")
                except Exception as e:
                    print(f"Error processing {dng_path}: {e}")


def find_dng_without_jpg(root_folder):
    """
    Recursively finds DNG files without a corresponding JPG file in the same directory.

    Args:
        root_folder (str): The directory to search in.
    """
    dng_without_jpg = []

    # Walk through the folder recursively
    for root, dirs, files in os.walk(root_folder):
        # Collect file names without extensions for quick comparison
        base_names = {os.path.splitext(f)[0] for f in files}

        # Check each DNG file
        for file in files:
            if file.lower().endswith(".dng"):
                base_name = os.path.splitext(file)[0]
                jpg_file = base_name + ".jpg"

                # Check if the JPG file is missing
                if jpg_file not in files and jpg_file.lower() not in files:
                    dng_without_jpg.append(os.path.join(root, file))

    # Print the list of DNG files without corresponding JPG files
    if dng_without_jpg:
        print("DNG files without corresponding JPG files:")
        for dng in dng_without_jpg:
            print(dng)
    else:
        print("All DNG files have matching JPG files.")


if __name__ == "__main__":
    # Path to your folder
    input_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_3_2024_12_12"

    # Run the function
    # convert_dng_to_jpg_recursive(input_folder)

    # check if there are DNG files without corresponding JPG files
    find_dng_without_jpg(input_folder)
