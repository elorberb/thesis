from src.datasets_and_annotations import annotation_handler
from src.segmentation.framework_handlers import detectron2_handler
from ultralytics.data.converter import convert_coco

if __name__ == "__main__":

    # prepare data from a single dataset to fine tune yolo v5
    segments_dataset_name = "etaylor/stigmas_dataset"
    dataset_folder_name = "etaylor_stigmas_dataset"
    release = "v0.3"
    saving_yaml_path = "/home/etaylor/code_projects/thesis/src/stigmas_detection/segmentation/yolo/yaml"

    # detectron2_handler.register_dataset(segments_dataset_name, release)

    convert_coco(
        labels_dir="/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/annotations",
        save_dir="/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/yolo",
        use_segments=True,
    )
