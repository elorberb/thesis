import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

#eval detectron2 imports
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from segments.utils import export_dataset
from src.annotation_handling.segmentsai_handler import SegmentsAIHandler
import os
import numpy as np
from datetime import datetime
import yaml
from matplotlib import pyplot as plt
import cv2
import torch

# Global variables
SEGMENTS_HANDLER = SegmentsAIHandler()
DETECTRON2_CHECKPOINT_BASE_PATH = "checkpoints/detectron2"

def print_version_info():
    torch_version = ".".join(torch.__version__.split(".")[:2])
    cuda_version = torch.version.cuda
    print("torch: ", torch_version, "; cuda: ", cuda_version)
    print("detectron2:", detectron2.__version__)


def convert_detectron2_to_segments_format(image, outputs):
    segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
    annotations = []
    counter = 1
    instances = outputs["instances"]
    for i in range(len(instances.pred_classes)):
        category_id = int(instances.pred_classes[i])
        instance_id = counter
        mask = instances.pred_masks[i].cpu()
        segmentation_bitmap[mask] = instance_id
        annotations.append(
            {"id": instance_id, "category_id": category_id}
        )
        counter += 1
    return segmentation_bitmap, annotations

def convert_segments_to_detectron2_format(dataset_name, release_version, export_format="coco-instance", output_dir="."):
    # get the dataset instance
    dataset = SEGMENTS_HANDLER.get_dataset_instance(dataset_name, version=release_version)
        
    # export the dataset - format is coco instance segmentation
    export_json_path, saved_images_path = export_dataset(dataset, export_format=export_format, export_folder=output_dir)
    return dataset, export_json_path, saved_images_path


class Detectron2Handler:
    
    def __init__(self, **kwargs):
        """
        Initializes the Detectron2Handler with the given keyword arguments.

        Keyword Arguments:
        - config_path (str): Path to the Detectron2 configuration file.
        - model_weights_url (str): URL or local path to the model weights.
        - train_dataset_name (str): Name of the train dataset.
        - train_release (str): Version of the train dataset.
        - test_dataset_name (str): Name of the test dataset.
        - test_release (str): Version of the test dataset.
        - export_format (str, optional): Format for exporting the dataset, default is 'coco-instance'.
        - input_mask_format (str, optional): Format of the input mask, default is 'bitmask'.
        - model_device (str, optional): Device to run the model on, default is 'cuda'.
        """
        
        setup_logger()

        # Extracting configuration settings from kwargs
        self.train_dataset_name = kwargs.get('train_dataset_name')
        train_release = kwargs.get('train_release')
        self.test_dataset_name = kwargs.get('test_dataset_name')
        test_release = kwargs.get('test_release')
        export_format = kwargs.get('export_format', 'coco-instance')
        input_mask_format = kwargs.get('input_mask_format', 'bitmask')
        model_device = kwargs.get('model_device', 'cuda')
        
        self.task_type = kwargs.get('task_type')
        self.model_type = kwargs.get('model_type')
        model_config_path = f"{self.task_type}/{self.model_type}.yaml"
                
        # Setting up the output directory
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_dir = f"{DETECTRON2_CHECKPOINT_BASE_PATH}/{self.task_type}/{self.model_type}/{timestamp}"

        # convert segments dataset to coco format
        self.train_dataset, train_export_json_path, train_saved_images_path = convert_segments_to_detectron2_format(
            dataset_name=self.train_dataset_name, 
            release_version=train_release, 
            export_format=export_format, 
            output_dir=output_dir
        )
        
        self.test_dataset, test_export_json_path, test_saved_images_path = convert_segments_to_detectron2_format(
            dataset_name=self.test_dataset_name, 
            release_version=test_release, 
            export_format=export_format, 
            output_dir=output_dir
        )
        # register the coco format datasets
        register_coco_instances("my_dataset_train", {}, train_export_json_path, train_saved_images_path)
        register_coco_instances("my_dataset_val", {}, test_export_json_path, test_saved_images_path)

        # get the metadata and dataset dicts
        self.train_metadata = MetadataCatalog.get("my_dataset_train")
        self.train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
        self.test_metadata = MetadataCatalog.get("my_dataset_val")
        self.test_dataset_dicts = DatasetCatalog.get("my_dataset_val")


    # Current setup for training the model with default values (I should work also on testing different values)
    def setup_training(
        self,
        score_thresh_test=0.5,
        num_workers=2,
        ims_per_batch=2,
        base_lr=0.00025,
        max_iter=1000,
        batch_size_per_image=512,
        num_classes=4,
    ):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        # If num_classes is not provided, calculate it from dataset.categories
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes # Currently 4 classes: trichome, clear, cloudy and amber - can use this if not working: len(self.dataset.categories) if num_classes is None else num_classes    
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)



    def train(self):
        # Training the model
        self.trainer.train()
        
        # Saving the model
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        self.predictor = DefaultPredictor(self.cfg)
        
        # Saving the configuration
        config_yaml_path = os.path.join(self.cfg.OUTPUT_DIR, "config.yaml")
        with open(config_yaml_path, 'w') as f:
            yaml.dump(self.cfg, f)
            
    
    def plot_train_samples(self, indices=None, scale=0.5):
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
        selected_samples = self.train_dataset_dicts if indices is None else [self.train_dataset_dicts[i] for i in indices]

        for d in selected_samples:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=self.train_metadata, scale=scale)
            out = visualizer.draw_dataset_dict(d)
            image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.show()
            

    def plot_test_predictions(self, indices=None, scale=0.5):
        
        selected_samples = self.test_dataset_dicts if indices is None else [self.test_dataset_dicts[i] for i in indices]

        for d in selected_samples:
            im = cv2.imread(d["file_name"])
            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=self.test_metadata, scale=scale, 
                        instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.show()


    def evaluate(self):
        """
        Evaluate the model on the specified dataset.

        Parameters:
        dataset_name (str): The name of the dataset to evaluate on.
        """
        # Set the dataset for testing
        evaluator = COCOEvaluator("my_dataset_val", output_dir=os.path.join(self.cfg.OUTPUT_DIR, "output"))
        val_loader = build_detection_test_loader(self.cfg, "my_dataset_val")
        print(inference_on_dataset(self.predictor.model, val_loader, evaluator))

        
    
    def load_checkpoint(self, checkpoint_path, config_yaml_path=None):
        """
        Load a specific model weights checkpoint.

        Parameters:
        checkpoint_path (str): The file path to the checkpoint.
        """
        # Update the model configuration to use the specified checkpoint
        self.cfg.MODEL.WEIGHTS = checkpoint_path
        
        # Load the configuration from the specified yaml file
        if config_yaml_path is not None:
            self.cfg.merge_from_file(config_yaml_path)

        # Re-initialize the model predictor with the updated configuration
        self.predictor = DefaultPredictor(self.cfg)



    def predict_image(self, image_path, plot_image=False):
        image = cv2.imread(image_path)
        outputs = self.predictor(image)
        if plot_image:
            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(image[:, :, ::-1], metadata=self.train_metadata)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)# Plot the image
            plt.imshow(image_rgb)
            plt.show()
        segmentation_bitmap, annotations = convert_detectron2_to_segments_format(
            image, outputs
        )
        return segmentation_bitmap, annotations
    
    # TODO: this code is not working well
    # TODO: need to fix it and find a different may to reuse code for detectron2 notebooks


# if __name__ == "__main__":

#     model_config = {
#     'task_type': 'COCO-InstanceSegmentation',
#     'model_type': 'mask_rcnn_R_50_FPN_3x',
#     'dataset_name': 'etaylor/cannabis_patches_all_images',
#     'release_version': "v0.2",
#     'export_format': "coco-instance",
#     'model_device': "cuda",
#     }

#     handler = Detectron2Handler(**model_config)

#     handler.setup_training()
#     handler.train() 
 