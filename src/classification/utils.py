from fastai.vision.all import *
from fastai.vision import *

    
def load_classification_model(model_path):
    """ Load your model from the specified path """
    model = load_learner(model_path)
    return model


def classify_objects(cut_images, classification_model):
    classification_results = {}
    for label, cut_image in cut_images.items():
        # Assume your classification model has a predict method
        classification_result = classification_model.predict(cut_image)
        classification_results[label] = classification_result[0]
    return classification_results