
from extra_deps.fast_labeling_workflow.fast_labeling_utils import train_model
from src.annotation_handling.segmentsai.segmentsai_handler import (
    setup_connection_segmentsai,
    visualize_dataset,
    create_dataset,
    upload_images_to_segments,
    generate_and_upload_predictions,
)
from segments import SegmentsDataset

def setup_globals():
    client = setup_connection_segmentsai()
    TRAIN_DATASET_NAME = 'etaylor/cannabis_patches_all_images'
    return client, TRAIN_DATASET_NAME

def upload_and_visualize_initial_data(client, TRAIN_DATASET_NAME):
    # Visualize the dataset
    visualize_dataset(client, TRAIN_DATASET_NAME)

def train_segmentation_model(client, TRAIN_DATASET_NAME):
    # Initialize a dataset from the release file
    release = client.get_release(TRAIN_DATASET_NAME, 'v0.1')
    dataset = SegmentsDataset(release)

    # Train an instance segmentation model on the dataset
    model = train_model(dataset)
    return model

def create_and_upload_images_to_new_dataset(client, image_name, week, zoom_type):
    name = f"Cannabis_patches_{week}_{zoom_type}_{image_name}"
    description = f"Cannabis patches week {week} zoom type {zoom_type} of image {image_name}."
    task_type = "segmentation-bitmap"
    task_attributes = {
        "format_version": "0.1",
        "categories": [{"name": "trichome", "id": 1}],
    }

    absolute_images_path = f"/sise/home/etaylor/images/processed_images/cannabis_patches/{week}/{zoom_type}/{image_name}"

    dataset_name_test = f"etaylor/{name}"

    # Create the dataset:
    dataset_test = create_dataset(client, name, description, task_type, task_attributes)
    print(dataset_test)

    # Upload the Images that are not annotated to the dataset
    upload_images_to_segments(client, dataset_name_test, absolute_images_path)
    return dataset_name_test

def create_release_and_upload_predictions(client, dataset_name_test, model):
    name = 'v0.1'
    description = 'upload predictions to dataset.'
    release = client.add_release(dataset_name_test, name, description)
    print(release)

    generate_and_upload_predictions(client, dataset_name_test, model, visualize_flag=True)

def delete_dataset(client, dataset_name):
    # Delete the dataset - ONLY IF YOU WANT TO DELETE THE DATASET!!!
    client.delete_dataset(dataset_name)

def main(visualize: bool = False):
    client, TRAIN_DATASET_NAME = setup_globals()
    if visualize:
        visualize_dataset(client, TRAIN_DATASET_NAME)
        
    model = train_segmentation_model(client, TRAIN_DATASET_NAME)
    
    # Example usage with one image and week, you can loop over multiple if needed
    image_name = "IMG_2145"
    week = "week9_3xzoom_regular"
    dataset_name_test = create_and_upload_images_to_new_dataset(client, image_name, week)
    
    create_release_and_upload_predictions(client, dataset_name_test, model)

    # Uncomment the following line if you need to delete a dataset
    # delete_dataset(client, "etaylor/trichomes_patches_week8_zoom3_regular_IMG_IMG_1809")

if __name__ == "__main__":
    main()
