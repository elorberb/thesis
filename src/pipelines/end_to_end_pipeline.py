from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from src.pipelines.preprocessing_pipeline import preprocess_single_image

EXPERIMENT_IMAGES_PATH = "/home/etaylor/images/assessing_cannabis_experiment_images"
GREENHOUSE_IMAGES_PATHS = {
    "day1": "/home/etaylor/images/assessing_cannabis_experiment_images/day_1_2024_05_30/moment_macro_lens",
    "day2": "/home/etaylor/images/assessing_cannabis_experiment_images/day_2_2024_06_03/Greenhouse",
    "day3": "/home/etaylor/images/assessing_cannabis_experiment_images/day_3_2024_06_06/greenhouse",
    # "day4": "",
    # "day5": "",
    # "day6": "",
    # "day7": "",
    # "day8": "",
}





# end to end pipe for predict the trichome dist given an image or a folder of images

# steps for the pipeline
# 1. Load the model
# 2. Load the image
# 3. Preprocess the image
# 4. Predict the image
#   4.1 Images are located in folders of a specific flower, we should predict for each image in the folder and for each patch the dist.
# 5. Save the results and image predictions

# load_flower_images
def load_flower_images():
    pass

def load_shooting_day_images():
    pass

# currently working with faster rcnn model
def load_model():
    model_details = {
        "model_name": "faster_rcnn_R_50_C4_1x",
        "checkpoint": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/model_final.pth",
        "yaml_file": "/home/etaylor/code_projects/thesis/checkpoints/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/29-04-2024_16-09-41/config.yaml"
    }
    
    cfg = get_cfg()
    # load config
    cfg.merge_from_file(model_details['yaml_file'])
    
    # load checkpoint
    cfg.MODEL.WEIGHTS = model_details['checkpoint']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # define predictor
    predictor = DefaultPredictor(cfg)
    
    return predictor


def preprocess_image(image_number, image_path):
    
        preprocessed_images = {}

        preprocessed_image = preprocess_single_image(image_path, image_number)
        preprocessed_images[image_number] = preprocessed_image
        return preprocessed_images

def predict_image():
    pass

def save_results():
    pass

def end_to_end_single_image():
    # load and preprocess image
    images_path = ""
    
    
    pass

def end_to_end_folder_images():
    pass

