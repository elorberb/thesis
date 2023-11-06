from extra_deps.fast_labeling_workflow.fast_labeling_utils import train_model
from src.annotation_handling.segmentsai_handler_old import (
    setup_connection_segmentsai,
    visualize_dataset,
    create_dataset,
    upload_images_to_segments,
    generate_and_upload_predictions,
)
from segments import SegmentsDataset
import config


def setup_globals():
    client = setup_connection_segmentsai()
    TRAIN_DATASET = "etaylor/cannabis_patches_all_images"
    return client, TRAIN_DATASET


def train_segmentation_model(client, TRAIN_DATASET_NAME):
    # Initialize a dataset from the release file
    release = client.get_release(TRAIN_DATASET_NAME, "v0.1")
    dataset = SegmentsDataset(release)

    # Train an instance segmentation model on the dataset
    model = train_model(dataset)
    return model


def create_new_test_dataset(client, image_name, week, zoom_type):
    name = f"Cannabis_patches_{week}_{zoom_type}_{image_name}"
    description = (
        f"Cannabis patches week {week} zoom type {zoom_type} of image {image_name}."
    )
    task_type = "segmentation-bitmap"
    task_attributes = {
        "format_version": "0.1",
        "categories": [{"name": "trichome", "id": 1}],
    }

    TEST_DATASET = f"etaylor/{name}"

    # Create the dataset:
    test_dataset_instance = create_dataset(client, name, description, task_type, task_attributes)
    print(test_dataset_instance)
    
    return TEST_DATASET


def create_release_and_upload_predictions(client, dataset_name_test, model):
    name = "v0.1"
    description = "upload predictions to dataset."
    release = client.add_release(dataset_name_test, name, description)
    print(release)

    generate_and_upload_predictions(
        client, dataset_name_test, model, visualize_flag=True
    )


def model_assist_label_pipeline(visualize: bool = False):
    client, TRAIN_DATASET_NAME = setup_globals()
    if visualize:
        visualize_dataset(client, TRAIN_DATASET_NAME)

    model = train_segmentation_model(client, TRAIN_DATASET_NAME)

    # Example usage with one image and week, you can loop over multiple if needed
    image_name = "IMG_2145"
    week = config.RAW_IMAGE_DIR['week9']
    zoom_type = config.RAW_IMAGE_DIR['3xr'] # currently only 3xr is supported
    TEST_DATASET = create_new_test_dataset(client, image_name, week, zoom_type)
    
    abs_images_path = f"{config.get_processed_cannabis_image_path(week, zoom_type)}/{image_name}"
    # Upload the Images that are not annotated to the dataset
    upload_images_to_segments(client, TEST_DATASET, abs_images_path)

    create_release_and_upload_predictions(client, TEST_DATASET, model)


if __name__ == "__main__":
    model_assist_label_pipeline()
