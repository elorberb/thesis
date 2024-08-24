import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

#eval detectron2 imports
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from src.datasets_and_annotations.segmentsai_handler import SegmentsAIHandler
from src.datasets_and_annotations import annotation_handler
import os
from matplotlib import pyplot as plt
import cv2
import torch
import csv
from skimage.measure import regionprops, label
import seaborn as sns
import pandas as pd

# Global variables
SEGMENTS_HANDLER = SegmentsAIHandler()
DETECTRON2_CHECKPOINT_BASE_PATH = "checkpoints/detectron2"

detectron2_detection_models = [
    "COCO-Detection/faster_rcnn_R_101_C4_3x",
    "COCO-Detection/faster_rcnn_R_101_DC5_3x",
    "COCO-Detection/faster_rcnn_R_101_FPN_3x",
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x",
    "COCO-Detection/retinanet_R_101_FPN_3x",
]

detectron2_segmentation_models = [
    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x",
    "COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x",
    "COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x",
    "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
    "Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml",
    "Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml",
    "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"
    "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml",
]


def print_version_info():
    torch_version = ".".join(torch.__version__.split(".")[:2])
    cuda_version = torch.version.cuda
    print("torch: ", torch_version, "; cuda: ", cuda_version)
    print("detectron2:", detectron2.__version__)


def imshow(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# Plot the image
    plt.imshow(image_rgb)
    plt.show()


def prepare_and_register_datasets(dataset_name_train, dataset_name_test, release_train, release_test):
    # Convert segments dataset to coco format for training dataset
    _, train_export_json_path, train_saved_images_path = annotation_handler.convert_segments_to_coco_format(
        dataset_name=dataset_name_train, 
        release_version=release_train, 
    )

    # Convert segments dataset to coco format for testing dataset
    _, test_export_json_path, test_saved_images_path = annotation_handler.convert_segments_to_coco_format(
        dataset_name=dataset_name_test, 
        release_version=release_test, 
    )

    # Register the coco format datasets
    register_coco_instances(dataset_name_train, {}, train_export_json_path, train_saved_images_path)
    register_coco_instances(dataset_name_test, {}, test_export_json_path, test_saved_images_path)

    # Get the metadata and dataset dicts
    metadata_train = MetadataCatalog.get(dataset_name_train)
    dataset_dicts_train = DatasetCatalog.get(dataset_name_train)
    metadata_test = MetadataCatalog.get(dataset_name_test)
    dataset_dicts_test = DatasetCatalog.get(dataset_name_test)

    return metadata_train, dataset_dicts_train, metadata_test, dataset_dicts_test


def plot_train_samples(dataset_dicts_train, metadata_train, indices=None, scale=0.5):
    """
    Plots samples based on specified indices.

    Parameters:
    - indices (list): List of specific indices of samples to plot.
    - scale (float): Scale factor for the visualizer.
    
    Example usage:
    model - Detectron2Handler(...)
    model.plot_samples(indices=[0, 2, 5]) # to plot images at specific indices
    model..plot_samples() # to plot all images
    """
    selected_samples = dataset_dicts_train if indices is None else [dataset_dicts_train[i] for i in indices]

    for d in selected_samples:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_train, scale=scale)
        out = visualizer.draw_dataset_dict(d)
        image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()


def plot_test_predictions(dataset_dicts_test, metadata_test, predictor, indices=None, scale=0.5):
    
    selected_samples = dataset_dicts_test if indices is None else [dataset_dicts_test[i] for i in indices]

    for d in selected_samples:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata_test, scale=scale, 
                    instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()


def evaluate_model_on_dataset(cfg, predictor):
    evaluator = COCOEvaluator("my_dataset_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_output"))
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    return inference_on_dataset(predictor.model, val_loader, evaluator)


def extract_object_info_to_csv(input_images_directory, output_csv_path, predictor, metadata):
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])

        for image_filename in os.listdir(input_images_directory):
            image_path = os.path.join(input_images_directory, image_filename)
            new_im = cv2.imread(image_path)
            outputs = predictor(new_im)
            mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
            class_labels = outputs["instances"].pred_classes.to("cpu").numpy()
            labeled_mask = label(mask)
            props = regionprops(labeled_mask)

            for i, prop in enumerate(props):
                object_number = i + 1
                area = prop.area
                centroid = prop.centroid
                bounding_box = prop.bbox
                class_label = class_labels[i] if i < len(class_labels) else 'Unknown'
                class_name = metadata.thing_classes[class_label] if class_label != 'Unknown' else 'Unknown'
                csvwriter.writerow([image_filename, class_name, object_number, area, centroid, bounding_box])

    return f"Object-level information saved to CSV file at {output_csv_path}"


def plot_class_statistics(output_csv_path, metadata_train):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(output_csv_path)

    # Get class names from train_metadata.thing_classes
    class_names = metadata_train.thing_classes

    # Calculate the average number of objects per image for each class
    avg_objects_per_class = df.groupby(["File Name", "Class Name"])["Object Number"].count().reset_index()
    avg_objects_per_class = avg_objects_per_class.groupby("Class Name")["Object Number"].mean().reset_index()

    # Plot: Average number of objects per image for each class
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Class Name", y="Object Number", data=avg_objects_per_class, ci=None, order=class_names)
    plt.xticks(rotation=45)
    plt.xlabel("Class Name")
    plt.ylabel("Average Number of Objects per Image")
    plt.title("Average Number of Objects per Image for Each Class")
    plt.tight_layout()
    plt.show()

    # Calculate the average area of objects for each class
    avg_area_per_class = df.groupby("Class Name")["Area"].mean().reset_index()

    # Plot: Average area of objects for each class
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Class Name", y="Area", data=avg_area_per_class, ci=None, order=class_names)
    plt.xticks(rotation=45)
    plt.xlabel("Class Name")
    plt.ylabel("Average Area of Objects")
    plt.title("Average Area of Objects for Each Class")
    plt.tight_layout()
    plt.show()