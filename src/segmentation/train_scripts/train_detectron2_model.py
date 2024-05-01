import argparse
import logging
import os
import torch
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from datetime import datetime
from src.segmentation.framework_handlers.detectron2_handler import prepare_and_register_datasets
import yaml

from src.segmentation.framework_handlers.detectron2_handler import evaluate_model_on_dataset
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

def setup():
    torch.cuda.empty_cache()  # Free up unutilized memory
    setup_logger()

def train_and_eval_detectron2(model):
    logger.info(f"Training and evaluating model {model}")
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    detectron2_models_path = "/home/etaylor/code_projects/thesis/checkpoints/detectron2"
    model_saving_path = os.path.join(detectron2_models_path, model, current_time)
    os.makedirs(model_saving_path, exist_ok=True)
    
    # dataset used for the training
    dataset_name_train = "etaylor/cannabis_patches_train_26-04-2024_15-44-44"
    dataset_name_test = "etaylor/cannabis_patches_test_26-04-2024_15-44-44"
    release_train = "v0.1"
    release_test = "v0.1"
    
    # Register and prepare datasets
    _, _, _, _ = prepare_and_register_datasets(
        dataset_name_train, dataset_name_test, release_train, release_test)
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_saving_path
    cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name_train,)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 18450 # 100 epochs
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # Default is 512.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Save the configuration to a config.yaml file
    config_yaml_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    logger.info(f"saving config to {config_yaml_path}")
    with open(config_yaml_path, 'w') as file:
        yaml.dump(cfg, file)
        
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator(dataset_name_test, output_dir=os.path.join(cfg.OUTPUT_DIR, dataset_name_test))
    val_loader = build_detection_test_loader(cfg, dataset_name_test)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    
    # Check if the directory exists
    output_dir = os.path.join(cfg.OUTPUT_DIR, os.path.basename(dataset_name_test))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    eval_results_saving_path = os.path.join(cfg.OUTPUT_DIR, os.path.basename(dataset_name_test), "evaluation_results.json")
    # Save the eval_results dictionary to a JSON file
    with open(eval_results_saving_path, 'w') as file:
        json.dump(eval_results, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Detectron2 model.")
    parser.add_argument("--model", required=True, help="Name of the model to train.")

    args = parser.parse_args()
    
    setup()
    train_and_eval_detectron2(args.model)
