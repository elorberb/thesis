import argparse
import logging
import torch
from ultralytics import YOLO, RTDETR, settings
import config
import os 

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def empty_cuda_cache():
    """Empty unused memory from CUDA."""
    torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache")

def setup_ultralytics_settings(model_name):
    """Setup settings for Ultralytics usage based on the configuration."""
    settings.update({
        'runs_dir': os.path.join(config.ULTRALYTICS_RUNS_DIR, model_name),
        'weights_dir': config.ULTRALYTICS_WEIGHTS_DIR,
        'datasets_dir': config.ULTRALYTICS_DATASETS_DIR
    })
    logger.info("Ultralytics settings updated:\n%s", settings)

def tune_model(config_yaml, model_checkpoint, epochs=100, imgsz=512):
    """Train the model using specified configurations."""
    ultralytics_checkpoint_paths = "/home/etaylor/code_projects/thesis/checkpoints/ultralytics"
    model_checkpoint_path = os.path.join(ultralytics_checkpoint_paths, model_checkpoint)
    

    if model_checkpoint == 'rtdetr-x.pt':
        model = RTDETR(model_checkpoint_path)
    else:
        model = YOLO(model_checkpoint_path)
    
    results = model.tune(data=config_yaml, epochs=epochs, imgsz=imgsz, batch=8, device=0)
    return model, results

def validate_model(model):
    """Validate the model and log the results."""
    valid_results = model.val()
    logger.info("Validation results:\n%s", valid_results)
    return valid_results

def main(args):
    """Main function to execute training and validation."""
    model_name = args.checkpoint.split('.')[0]
    empty_cuda_cache()
    setup_ultralytics_settings(model_name)

    logger.info(f"Starting training of Ultralytics model {args.checkpoint}")
    model, results = tune_model(args.config, args.checkpoint, args.epochs, args.imgsz)
    logger.info("Training results:\n%s", results)

    validate_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Validate Ultralytics models.")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--imgsz', type=int, default=512, help='Image size for training')

    args = parser.parse_args()
    main(args)
