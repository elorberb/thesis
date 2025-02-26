import os


def count_jpg_images(base_dir):
    total_images = 0

    # List all items in the base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        # Check if it's a directory and its name starts with "day_"
        if os.path.isdir(item_path) and item.startswith("day_"):
            # Walk through the directory recursively
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    # Check if the file is a JPG (case-insensitive)
                    if file.lower().endswith(".jpg"):
                        total_images += 1
    return total_images


def count_images_in_folder(folder_path):
    """
    Recursively count .jpg images in the given folder.
    """
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                count += 1
    return count


def count_images_in_each_subfolder(base_folder):
    """
    For each immediate subfolder of base_folder, count the number of JPG images
    (recursively) and print the folder name along with the count.
    """
    # List all immediate items in the base folder
    with os.scandir(base_folder) as entries:
        for entry in entries:
            if entry.is_dir():
                folder_path = entry.path
                image_count = count_images_in_folder(folder_path)
                print(f"Folder: {entry.name} - {image_count} image(s)")


if __name__ == "__main__":
    base_folder = (
        "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images"
    )
    jpg_count = count_jpg_images(base_folder)
    print(f"Total number of JPG images in experiment 1: {jpg_count}")

    base_folder = (
        "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images"
    )
    jpg_count = count_jpg_images(base_folder)
    print(f"Total number of JPG images in experiment 2: {jpg_count}")

    base_folder = "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/good_quality"
    count_images_in_each_subfolder(base_folder)

    # count how many images in this path: /sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/stigmas_images_flat
    folder_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/stigmas_images_flat"
    image_count = count_images_in_folder(folder_path)
    print(f"Folder: stigmas dataset - {image_count} image(s)")
