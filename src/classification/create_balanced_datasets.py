import os
import shutil
import random


def print_image_distribution(dataset_path):
    """
    Prints the number of images per class in the given dataset path.

    Args:
        dataset_path (str): The path containing subdirectories for each class.
    """
    if not os.path.exists(dataset_path):
        print(f"The path '{dataset_path}' does not exist.")
        return

    print(f"--- Image Distribution in '{dataset_path}' ---")
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory
            class_count = len(os.listdir(class_path))
            print(f"Class '{class_name}': {class_count} images")
    print()


def create_balanced_train_test_split(
    original_path, balanced_path, dataset_num, target_count=200
):
    """
    Creates a balanced training dataset with the specified number of images per class,
    and uses the remaining images for a test set.

    Args:
        original_path (str): Path to the original dataset containing 'train' directory.
        balanced_path (str): Path where the balanced dataset will be saved.
        dataset_num (int): Identifier for the dataset version being created.
        target_count (int): Number of images per class in the balanced training set.
    """
    train_path = os.path.join(original_path, "train")
    balanced_train_path = os.path.join(
        balanced_path, f"train_set_{dataset_num}", "train"
    )
    test_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "test")

    # Ensure the balanced dataset directories exist
    os.makedirs(balanced_train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    def split_images_to_train_test(
        source_class_path, target_train_path, target_test_path, target_count=200
    ):
        """Splits images into a balanced training set and a test set."""
        images = os.listdir(source_class_path)
        random.shuffle(images)

        # Select target_count images for training
        train_images = images[:target_count]
        test_images = images[target_count:]  # Remaining images for test set

        # Copy selected images to the target training directory
        for i, img_name in enumerate(train_images):
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(target_train_path, f"{i}_{img_name}")
            shutil.copy(src_img_path, dest_img_path)

        # Copy remaining images to the test directory
        for img_name in test_images:
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(target_test_path, img_name)
            shutil.copy(src_img_path, dest_img_path)

    # Process each class in the train directory
    classes = os.listdir(train_path)
    for class_name in classes:
        source_class_path = os.path.join(train_path, class_name)
        target_class_train_path = os.path.join(balanced_train_path, class_name)
        target_class_test_path = os.path.join(test_path, class_name)

        os.makedirs(target_class_train_path, exist_ok=True)
        os.makedirs(target_class_test_path, exist_ok=True)

        print(f"Creating balanced dataset {dataset_num} - Class {class_name}")
        split_images_to_train_test(
            source_class_path,
            target_class_train_path,
            target_class_test_path,
            target_count,
        )


def create_balanced_split(original_path, balanced_path, dataset_num, target_count=200):
    """
    Creates a balanced dataset split with specified number of images per class for training,
    and copies remaining images for validation and test sets. Copies existing validation images
    from the original validation dataset.

    Args:
        original_path (str): Path to the original dataset containing 'train' and 'val' directories.
        balanced_path (str): Path where the balanced datasets will be saved.
        dataset_num (int): Identifier for the dataset version being created.
        target_count (int): Number of images per class in the balanced training set.
    """
    train_path = os.path.join(original_path, "train")
    original_val_path = os.path.join(original_path, "val")
    balanced_train_path = os.path.join(
        balanced_path, f"train_set_{dataset_num}", "train"
    )
    new_val_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "val")
    test_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "test")

    # Ensure the balanced dataset directories exist
    os.makedirs(balanced_train_path, exist_ok=True)
    os.makedirs(new_val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    def balance_class_images_to_train_val_test(
        source_class_path, target_class_path, val_class_path, target_count=200
    ):
        """Balance the number of images for a class by creating a split for training, val, and test sets."""
        images = os.listdir(source_class_path)
        random.shuffle(images)

        # Select target_count images for training
        train_images = images[:target_count]
        val_images = images[target_count:]  # Remaining images for val/test

        # Copy selected images to the target training directory
        for i, img_name in enumerate(train_images):
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(target_class_path, f"{i}_{img_name}")
            shutil.copy(src_img_path, dest_img_path)

        # Split remaining images between val and test (50% each)
        val_split = len(val_images) // 2
        for img_name in val_images[:val_split]:
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(val_class_path, img_name)
            shutil.copy(src_img_path, dest_img_path)

        # Ensure each class has a subfolder in the test directory
        class_test_path = os.path.join(
            target_class_path, os.path.basename(source_class_path)
        )
        os.makedirs(class_test_path, exist_ok=True)

        for img_name in val_images[val_split:]:
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(
                class_test_path, img_name
            )  # Save remaining images in class-specific subfolder
            shutil.copy(src_img_path, dest_img_path)

    # Process each class in the train directory
    classes = os.listdir(train_path)
    for class_name in classes:
        source_class_path = os.path.join(train_path, class_name)
        target_class_path = os.path.join(balanced_train_path, class_name)
        val_class_path = os.path.join(new_val_path, class_name)

        os.makedirs(target_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

        print(f"Creating balanced dataset {dataset_num} - Class {class_name}")
        balance_class_images_to_train_val_test(
            source_class_path, target_class_path, val_class_path
        )

    # Copy the existing validation images to the new val directory
    for class_name in os.listdir(original_val_path):
        original_class_path = os.path.join(original_val_path, class_name)
        new_val_class_path = os.path.join(new_val_path, class_name)
        os.makedirs(new_val_class_path, exist_ok=True)

        for img_name in os.listdir(original_class_path):
            src_img_path = os.path.join(original_class_path, img_name)
            dest_img_path = os.path.join(new_val_class_path, img_name)
            shutil.copy(src_img_path, dest_img_path)


def split_val_to_val_test(balanced_path, dataset_num, split_ratio=0.5):
    """
    Splits the existing validation set into new val and test sets for the specified dataset.

    Args:
        balanced_path (str): Base path where balanced datasets are stored.
        dataset_num (int): Identifier for the dataset version to process.
        split_ratio (float): Ratio of images to keep in the validation set (remaining goes to test).
    """
    val_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "val")
    test_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "test")

    for class_name in os.listdir(val_path):
        class_val_path = os.path.join(val_path, class_name)
        class_test_path = os.path.join(test_path, class_name)

        os.makedirs(class_test_path, exist_ok=True)

        images = os.listdir(class_val_path)
        random.shuffle(images)

        # Split images according to split_ratio
        split_point = int(len(images) * split_ratio)
        new_val_images = images[:split_point]
        new_test_images = images[split_point:]

        # Move images to new test set
        for img_name in new_test_images:
            src_img_path = os.path.join(class_val_path, img_name)
            dest_img_path = os.path.join(class_test_path, img_name)
            shutil.move(src_img_path, dest_img_path)

    print(f"Validation set for dataset {dataset_num} split into new val and test sets.")


def create_balanced_test_set(source_dir, dest_dir, num_samples_per_class=26):
    """
    Creates a balanced test set by sampling a specified number of images from each class.

    Parameters:
    - source_dir (str): Path to the original test dataset directory.
    - dest_dir (str): Path to the destination directory for the balanced test set.
    - num_samples_per_class (int): Number of images to sample per class. Default is 26.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over each class directory in the source directory
    for class_name in os.listdir(source_dir):
        class_source_path = os.path.join(source_dir, class_name)
        class_dest_path = os.path.join(dest_dir, class_name)

        # Check if it's a directory
        if os.path.isdir(class_source_path):
            # Ensure the class directory exists in the destination
            os.makedirs(class_dest_path, exist_ok=True)

            # List all image files in the class directory
            image_files = [
                f
                for f in os.listdir(class_source_path)
                if os.path.isfile(os.path.join(class_source_path, f))
            ]

            # Check if there are enough images to sample
            if len(image_files) < num_samples_per_class:
                print(
                    f"Warning: Class '{class_name}' has only {len(image_files)} images. All will be copied."
                )
                sampled_files = image_files
            else:
                # Randomly sample the specified number of images
                sampled_files = random.sample(image_files, num_samples_per_class)

            # Copy the sampled files to the destination directory
            for file_name in sampled_files:
                src_file = os.path.join(class_source_path, file_name)
                dest_file = os.path.join(class_dest_path, file_name)
                shutil.copy2(src_file, dest_file)

            print(f"Copied {len(sampled_files)} images for class '{class_name}'.")


if __name__ == "__main__":

    # ---------------- Create multiple balanced datasets ----------------
    # Paths for original and balanced datasets
    # original_path = "/home/etaylor/code_projects/thesis/classification_datasets/blur_classification/blur_classification_datasets/all_classes"
    balanced_path = "/home/etaylor/code_projects/thesis/classification_datasets/blur_classification/blur_classification_datasets/all_classes/balanced_datasets"

    # # Generate multiple balanced datasets
    # for dataset_num in range(1, 6):
    #     create_balanced_train_test_split(
    #         original_path, balanced_path, dataset_num, target_count=400
    #     )

    # print("Multiple balanced datasets created successfully!")

#     # # ---------------- Split validation set into new val and test sets ----------------
#     # # Run the split function for each dataset
#     # for dataset_num in range(1, 6):
#     #     split_val_to_val_test(balanced_path, dataset_num, split_ratio=0.5)

#     # ---------------- Check how many images are in each class ----------------

    # for dataset_num in range(1, 6):
    #     dataset_path = os.path.join(balanced_path, f"train_set_{dataset_num}")

    #     # get the train, val and test datasets
    #     train_path = os.path.join(dataset_path, "train")
    #     test_path = os.path.join(dataset_path, "test")

    #     print(f"Dataset {dataset_num}")
    #     print_image_distribution(train_path)
    #     print_image_distribution(test_path)

#     print("Done!")

#     # # count the files that are not folders in this path /home/etaylor/code_projects/thesis/segments/etaylor_cannabis_patches_train_26-04-2024_15-44-44/ground_truth_trichomes_datasets/trichome_dataset_125_good_quality/balanced_datasets/train_set_1/test"
#     # count = 0
#     # for file in os.listdir("/home/etaylor/code_projects/thesis/segments/etaylor_cannabis_patches_train_26-04-2024_15-44-44/ground_truth_trichomes_datasets/trichome_dataset_125_good_quality/balanced_datasets/train_set_1/test"):
#     #     if not os.path.isdir(file):
#     #         count += 1
#     # print(count)

#     # lets check if the files that are not in the class folders are in the test folder


# ---------------- Create balanced test set ----------------
    for dataset_num in range(1, 6):
        dataset_path = os.path.join(balanced_path, f"train_set_{dataset_num}")

        # get the train, val and test datasets
        test_balanced_path = os.path.join(dataset_path, "test_balanced")
        test_path = os.path.join(dataset_path, "test")
        
        create_balanced_test_set(test_path, test_balanced_path, num_samples_per_class=196)
        print_image_distribution(test_balanced_path)
