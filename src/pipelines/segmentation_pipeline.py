from src.segmentation.framework_handlers.detectron2_handler import Detectron2Handler
from src.data_preparation.image_loader import read_images_and_names


# TODO: Implement the following functions
def predict_masks(images, detectron_handler):
    pass


def save_masks(masks, output_folder):
    pass


def segmentation_pipeline(input_folder, output_folder, detectron_handler):
    images_names = read_images_and_names(input_folder)
    masks = predict_masks(images_names, detectron_handler)
    save_masks(masks, output_folder)
    print("Segmentation pipeline process completed.")
    
    
if __name__ == "__main__":
    model_config = {
    'task_type': 'COCO-InstanceSegmentation',
    'model_type': 'mask_rcnn_R_50_FPN_3x',
    'dataset_name': 'etaylor/cannabis_patches_all_images',
    'release_version': "v0.2",
    'export_format': "coco-instance",
    'model_device': "cuda",
    }

    model = Detectron2Handler(**model_config)
    input_folder = '/path/to/preprocessed/images'
    output_folder = '/path/to/save/masks'

    segmentation_pipeline(input_folder, output_folder, model)
