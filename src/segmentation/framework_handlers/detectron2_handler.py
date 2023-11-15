from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from segments.utils import export_dataset
from src.annotation_handling.segmentsai_handler import SegmentsAIHandler
import os
import numpy as np
from datetime import datetime


SEGMENTS_HANDLER = SegmentsAIHandler()
DETECTRON2_CHECKPOINT_BASE_PATH = "checkpoints/detectron2"



def convert_detectron2_to_segments_format(image, outputs):
    segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
    annotations = []
    counter = 1
    instances = outputs["instances"]
    for i in range(len(instances.pred_classes)):
        instance_id = counter
        mask = instances.pred_masks[i].cpu()
        segmentation_bitmap[mask] = instance_id
        trichome_annotation_category_id = (
            1  # assuming a single category for simplicity
        )
        annotations.append(
            {"id": instance_id, "category_id": trichome_annotation_category_id}
        )
        counter += 1
    return segmentation_bitmap, annotations

def convert_segments_to_detectron2_format(
    dataset_name, release_version, export_format="coco-instance", output_dir="."
):
    # get the dataset instance
    dataset = SEGMENTS_HANDLER.get_dataset_instance(dataset_name, version=release_version, output_dir=output_dir)
        
    # export the dataset
    export_file, image_dir = export_dataset(dataset, export_format=export_format, export_folder=output_dir)
    return dataset, export_file, image_dir


class Detectron2Handler:
    
    def __init__(self, **kwargs):
        """
        Initializes the Detectron2Handler with the given keyword arguments.

        Keyword Arguments:
        - config_path (str): Path to the Detectron2 configuration file.
        - model_weights_url (str): URL or local path to the model weights.
        - dataset_name (str): Name of the dataset to be used.
        - release_version (str): Version of the dataset.
        - export_format (str, optional): Format for exporting the dataset, default is 'coco-instance'.
        - input_mask_format (str, optional): Format of the input mask, default is 'bitmask'.
        - model_device (str, optional): Device to run the model on, default is 'cuda'.
        """

        # Extracting configuration settings from kwargs
        train_dataset_name = kwargs.get('dataset_name')
        release_version = kwargs.get('release_version')
        export_format = kwargs.get('export_format', 'coco-instance')
        input_mask_format = kwargs.get('input_mask_format', 'bitmask')
        model_device = kwargs.get('model_device', 'cuda')
        
        self.task_type = kwargs.get('task_type')
        self.model_type = kwargs.get('model_type')
        model_config_path = f"{self.task_type}/{self.model_type}.yaml"
        self.cfg = get_cfg()
                
        # Setting up the output directory
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_dir = f"{DETECTRON2_CHECKPOINT_BASE_PATH}/{self.task_type}/{self.model_type}/{timestamp}"
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Rest of the initialization code
        self.dataset, export_file, image_dir = convert_segments_to_detectron2_format(
            dataset_name=train_dataset_name, 
            release_version=release_version, 
            export_format=export_format, 
            output_dir=output_dir
        )
        try:
            register_coco_instances(train_dataset_name, {}, export_file, image_dir)
        except Exception as e:
            print(f"Dataset registration failed: {e}")
        MetadataCatalog.get(train_dataset_name).set(
            thing_classes=[c.name for c in self.dataset.categories]
        )
        segments_metadata = MetadataCatalog.get(train_dataset_name)
        print(segments_metadata)
        
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config_path))
        self.cfg.DATASETS.TRAIN = train_dataset_name
        self.cfg.DATASETS.TEST = ()
        self.cfg.INPUT.MASK_FORMAT = input_mask_format
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config_path)
        self.cfg.MODEL.DEVICE = model_device
        self.predictor = DefaultPredictor(self.cfg)


    # Current setup for training the model with default values (I should work also on testing different values)
    def setup_training(
        self,
        score_thresh_test=0.7,
        num_workers=2,
        ims_per_batch=2,
        base_lr=0.00025,
        max_iter=300,
        batch_size_per_image=128,
        num_classes=None,
    ):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        # If num_classes is not provided, calculate it from dataset.categories
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
            num_classes if num_classes is not None else len(self.dataset.categories)
        )



    def train(self):
        # Training the model
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Saving the model
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        
        
    # TODO: create a function to eval the model on the test set
    def evaluate(self, dataset_name):
        """
        Evaluate the model on the specified dataset.

        Parameters:
        dataset_name (str): The name of the dataset to evaluate on.
        """
        # Set the dataset for testing
        self.cfg.DATASETS.TEST = (dataset_name, )

        


    def predict_image(self, image):
        outputs = self.predictor(image)
        segmentation_bitmap, annotations = convert_detectron2_to_segments_format(
            image, outputs
        )
        return segmentation_bitmap, annotations


if __name__ == "__main__":

    config = {
    'task_type': 'COCO-InstanceSegmentation',
    'model_type': 'mask_rcnn_R_50_FPN_3x',
    'dataset_name': 'etaylor/cannabis_patches_all_images',
    'export_format': "coco-instance",
    'model_device': "cuda",
    'release_version': "v0.2",
}

    handler = Detectron2Handler(**config)

    handler.setup_training()
    handler.train() 
 
 #TODO: add load model from checkpoint function
 #TODO: think of an efficient way to save the models checkpoints
    # TODO: idea: framework/task/model_name(using the names from the model_zoo)/version(datetime)  
    # TODO: example: detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/2021-05-05_12:00:00