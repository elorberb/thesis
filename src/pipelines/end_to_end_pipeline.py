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


pipe_psudo = """
end to end pipe for predict the trichome dist given an image or a folder of images.

to perform the end to end pipeline, we can choose one of two methods:
1. use my pre-process pipe for the images and then run rest of the pipe
2. use sahi package for the cutting and inference.

steps after pre-process:
1. I should use the detection model (Faster RCNN) for the object detection part.
2. I should extract the trichomes detected for the classification part
- better to do this is a temp folder that would be deleted after I perform the classification.
3. I should use the classification model (Alexnet) for the classification part and save the labels of each trichome.
4. I should calc the aggregated score for each full image and save it to json.
5. If using Sahi - I should feedback the trichomes labels to the detected object in order to plot sample of the results.

Improvements:
- maybe it is possible to perform the classification without saving the trichomes images (just to point for the bbox values of the images patches).
- Improve the sahi inference by filtering patches that are non relevant for the detection (blurry areas for example).
- maybe it is possible to use sahi with two models, one for detection and one for the classification, instead of extracting trichomes and then classify them.


Next steps:
- download all the images data to the cluster.
- create the python env that can run both detectron2 model and fastai model.
- create a notebook for the end to end pipeline - test weather it is possible to perform it for a single image.
- extend to a full image
- extend to a folder of images (which represent a flower)

"""

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

