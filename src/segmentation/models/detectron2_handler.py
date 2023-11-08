from dotenv import load_dotenv
load_dotenv()


from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from segments.utils import export_dataset
import os
import numpy as np


class Detectron2Handler:
    
    def __init__(self, config_path, model_weights_url, score_thresh_test):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = model_weights_url
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        self.cfg.MODEL.DEVICE = 'cuda'
        self.predictor = DefaultPredictor(self.cfg)


    def register_dataset(self, dataset):
        
        export_file, image_dir = export_dataset(dataset, export_format="coco-instance")
        
        # Register it as a COCO dataset in the Detectron2 framework
        try:
            register_coco_instances("my_dataset", {}, export_file, image_dir)
        except:
            print("Dataset was already registered")
        dataset_dicts = load_coco_json(export_file, image_dir)
        MetadataCatalog.get("my_dataset").set(
            thing_classes=[c.name for c in dataset.categories]
        )
        segments_metadata = MetadataCatalog.get("my_dataset")
        print(segments_metadata)


    def setup_training(self, dataset):
        self.cfg.DATASETS.TRAIN = ("my_dataset",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.INPUT.MASK_FORMAT = "bitmask"
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 2  # 4
        self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 300  
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128  # faster, and good enough for this toy dataset (default: 512)
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataset.categories)  # number of categories

        # Additional setup specific to training

    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        

    def convert_to_segments_format(self, image, outputs):
        segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
        annotations = []
        counter = 1
        instances = outputs["instances"]
        for i in range(len(instances.pred_classes)):
            instance_id = counter
            mask = instances.pred_masks[i].cpu()
            segmentation_bitmap[mask] = instance_id
            trichome_annotation_category_id = 1  # assuming a single category for simplicity
            annotations.append(
                {"id": instance_id, "category_id": trichome_annotation_category_id}
            )
            counter += 1
        return segmentation_bitmap, annotations
    

    def predict_image(self, image):
        outputs = self.predictor(image)
        segmentation_bitmap, annotations = self._convert_to_segments_format(image, outputs)
        return segmentation_bitmap, annotations


if __name__ == "__main__":
    # Usage Example
    model_config_path = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    model_weights = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 

    detectron_handler = Detectron2Handler(
        config_path=model_config_path,
        model_weights_url=model_weights,
        score_thresh_test=0.7,    
        )
    # Assuming dataset_name, coco_json, and image_dir are provided from the SegmentsAI export
    detectron_handler.register_dataset(dataset_name, coco_json, image_dir)
    detectron_handler.setup_training(dataset_name)
    detectron_handler.train()
    model = detectron_handler.load_model(score_thresh_test=0.7)
