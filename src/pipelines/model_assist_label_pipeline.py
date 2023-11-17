""" 
Model Assist Label Pipeline Stages:

    Sets up the Segments.ai API connection.
    Trains the model on the specified dataset.
    Creates a new dataset for testing from a specific image patches.
    Uploads images to the new dataset.
    Creates a release and uploads predictions as annotations to the dataset.
"""

from extra_deps.fast_labeling_workflow.fast_labeling_utils import train_model
from src.annotation_handling.segmentsai_handler import SegmentsAIHandler
from segments import SegmentsDataset
import config

segmentsai_handler = SegmentsAIHandler()



def train_segmentation_model(train_dataset_name):
    # Initialize a dataset from the release file
    release = segmentsai_handler.client.get_release(train_dataset_name, "v0.2")
    dataset = SegmentsDataset(release)

    # Train an instance segmentation model on the dataset
    model = train_model(dataset)
    return model


def create_new_test_dataset(image_name, week, zoom_type, single_category=True):
    dataset_name = f"cannabis_patches_{week}_{zoom_type}_{image_name}"
    description = (
        f"cannabis patches week={week} zoom_type={zoom_type} of image={image_name}."
    )
    task_type = "segmentation-bitmap"
    if single_category:
        task_attributes = {
            "format_version": "0.1",
            "categories": [{"name": "trichome", "id": 1, "color": [65,117,5]}],
        }
    else:
        task_attributes = {
            "format_version": "0.1",
            "categories": [{"name": "trichome", "id": 1, "color": [65,117,5]},
                        {"name": "clear", "id": 2, "color": [155,155,155]},
                        {"name": "cloudy", "id": 3, "color": [255,255,255]},
                        {"name": "amber", "id": 4, "color": [245,166,35]}]
        }

    TEST_DATASET = f"etaylor/{dataset_name}"

    # Create the dataset:
    test_dataset_instance = segmentsai_handler.create_new_dataset(dataset_name, description, task_type, task_attributes)
    print(test_dataset_instance)
    
    return TEST_DATASET


def upload_predictions(release, model):
    print(f"release={release}")
    dataset = SegmentsDataset(release)


    for sample in dataset:
        # Generate label predictions
        image = sample["image"]
        segmentation_bitmap, annotations = model(image)
        segmentsai_handler.upload_annotation_for_sample(sample['uuid'], segmentation_bitmap, annotations)


def model_assist_label_pipeline(
    image_name: str, week_key: str, zoom_type_key: str, visualize: bool = False
):
    
    train_dataset_name = "etaylor/cannabis_patches_all_images"

    if visualize:
        segmentsai_handler.visualize_dataset(train_dataset_name)

    model = train_segmentation_model(train_dataset_name)
    
    # Create a new test dataset for the specified image, week, and zoom type
    test_dataset = create_new_test_dataset(image_name, config.WEEKS_DIR[week_key], config.ZOOM_TYPES_DIR[zoom_type_key])
    # Get the absolute path to the processed image
    abs_images_path = f"{config.get_processed_cannabis_image_path(week_key, zoom_type_key)}/{image_name}"
    
    # Upload the images that are not annotated to the dataset
    segmentsai_handler.upload_images(test_dataset, abs_images_path)
    
    release_name = "v0.1"
    description = "upload predictions to dataset."
    segmentsai_handler.client.add_release(test_dataset, release_name, description)
    test_release = segmentsai_handler.client.get_release(test_dataset, "v0.1")
    # Create a release and upload predictions to the platform
    upload_predictions(test_release, model)



if __name__ == "__main__":
    # TODO: THIS SCRIPT DOES NOT WORK BECAUSE OF A PROBLEM WITH THE RELEASE OF SEGMENTS.AI - to run this code run it from here: src/annotation_handling/notebooks/model_assisted_labeling.ipynb
    image_name_param = "IMG_2129"
    week_param = 'week9'
    zoom_type_param = '3xr'
    model_assist_label_pipeline(image_name_param, week_param, zoom_type_param, visualize=True)
