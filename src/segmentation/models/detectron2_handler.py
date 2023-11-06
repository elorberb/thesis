import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

import os


class Detectron2Handler:
    
    def __init__(self, config_path, model_weights_url, output_dir):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = model_weights_url
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        # Additional configuration setup as required

    def register_dataset(self, dataset_name, coco_json, image_dir):
        try:
            register_coco_instances(dataset_name, {}, coco_json, image_dir)
            MetadataCatalog.get(dataset_name).set(thing_classes=[c.name for c in dataset.categories])
            print(MetadataCatalog.get(dataset_name))
        except ValueError:
            print(f"Dataset {dataset_name} was already registered")

    def setup_training(self, dataset_name):
        self.cfg.DATASETS.TRAIN = (dataset_name,)
        # Additional setup specific to training

    def train(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def load_model(self, score_thresh_test):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        predictor = DefaultPredictor(self.cfg)
        return Model(predictor)




# Usage Example
model_config_path = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
model_weights = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 

detectron_handler = DetectronHandler(
    config_path=model_config_path,
    model_weights_url=model_weights,
    output_dir="path_to_output_dir"
)
# Assuming dataset_name, coco_json, and image_dir are provided from the SegmentsAI export
detectron_handler.register_dataset(dataset_name, coco_json, image_dir)
detectron_handler.setup_training(dataset_name)
detectron_handler.train()
model = detectron_handler.load_model(score_thresh_test=0.7)
