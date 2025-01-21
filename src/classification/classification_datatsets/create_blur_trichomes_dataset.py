import cv2
import os
import numpy as np
import random
import shutil


# Blur application functions
def apply_blur(image, blur_type="gaussian", intensity=5):
    if blur_type == "gaussian":
        return cv2.GaussianBlur(image, (intensity, intensity), 0)
    elif blur_type == "motion":
        kernel_size = intensity
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(image, -1, kernel)
    elif blur_type == "median":
        return cv2.medianBlur(image, intensity)
    elif blur_type == "bilateral":
        return cv2.bilateralFilter(image, intensity, 75, 75)
    else:
        raise ValueError("Invalid blur type specified.")


# Create blur vs good quality for each class
def create_blur_vs_quality(
    input_folder, output_folder, blur_types, intensity_range=(5, 15), blur_fraction=0.5
):
    for trichome_class in ["clear", "cloudy", "amber"]:
        class_path = os.path.join(input_folder, trichome_class)
        blur_output_path = os.path.join(
            output_folder, "dataset_1", trichome_class, "blur"
        )
        quality_output_path = os.path.join(
            output_folder, "dataset_1", trichome_class, "good_quality"
        )

        os.makedirs(blur_output_path, exist_ok=True)
        os.makedirs(quality_output_path, exist_ok=True)

        files = os.listdir(class_path)
        selected_files = random.sample(files, int(len(files) * blur_fraction))

        # Copy good quality images
        for file in files:
            shutil.copy(
                os.path.join(class_path, file), os.path.join(quality_output_path, file)
            )

        # Create and save blurred versions
        for file in selected_files:
            img_path = os.path.join(class_path, file)
            image = cv2.imread(img_path)
            for blur_type in blur_types:
                intensity = random.randint(*intensity_range)
                blurred_img = apply_blur(image, blur_type, intensity)
                cv2.imwrite(
                    os.path.join(blur_output_path, f"{blur_type}_{intensity}_{file}"),
                    blurred_img,
                )


# Create combined good quality dataset
def create_combined_good_quality(input_folder, output_folder):
    combined_output_path = os.path.join(output_folder, "dataset_2", "good_quality")
    os.makedirs(combined_output_path, exist_ok=True)

    for trichome_class in ["clear", "cloudy", "amber"]:
        class_path = os.path.join(input_folder, trichome_class)
        files = os.listdir(class_path)

        # Copy all good quality images to the combined dataset
        for file in files:
            shutil.copy(
                os.path.join(class_path, file),
                os.path.join(combined_output_path, f"{trichome_class}_{file}"),
            )


if __name__ == "main":
    
    # Path configuration
    good_quality_path = "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/good_quality"
    output_base = (
        "/home/etaylor/code_projects/thesis/classification_datasets/output_dataset"
    )
    
    # Paths and configurations
    blur_algorithms = ["gaussian", "motion", "median"]

    # Run dataset creation
    create_blur_vs_quality(good_quality_path, output_base, blur_algorithms)
    create_combined_good_quality(good_quality_path, output_base)

    print("Datasets created successfully!")
