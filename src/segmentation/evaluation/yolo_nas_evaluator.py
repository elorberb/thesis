from src.segmentation.evaluation.ultralytics_evaluator import UltralyticsEvaluator
import re
import os

class YoloNasEvaluator(UltralyticsEvaluator):
    def __init__(self, num_classes, image_size):
        super().__init__(num_classes, image_size)



    def parse_model_outputs(self, outputs):
        """ Parse the model outputs to extract bounding boxes, confidence scores, and class IDs.
        
        Args:
            outputs (np.array): Array of bounding box coordinates [x_min, y_min, x_max, y_max].
            confidences (np.array): Array of confidence scores corresponding to each bbox.
        
        Returns:
            list: A list of dictionaries, each containing bounding box details and class ID.
        """
        # get predictions
        bboxes = outputs.prediction.bboxes_xyxy
        confidences = outputs.prediction.confidence
        class_ids = outputs.prediction.labels
        
        parsed_outputs = []
        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
            x_min, y_min, x_max, y_max = bbox
            parsed_outputs.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'score': confidence,
                'class_id': int(class_id)
            })
        return parsed_outputs

    
    
    def predict_and_parse_image_patches(self, image_number, images_directory, predictor):
        """
        Process all patches of a given image, predict on them, and collect outputs in the format of the parsed model output.
        Args:
            image_number (str): The base image number, e.g., 'IMG_2157'.
            images_directory (str): The directory path where image patches are stored.
            predictor (callable): The predictor object to use for making predictions.
        Returns:
            dict: A dictionary containing the patch file names as keys and prediction outputs as values.
        """
        parsed_outputs_by_patch = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.(jpg|png)$")

        for file_name in os.listdir(images_directory):
            if pattern.match(file_name):
                print(f"Processing patch: {file_name}")
                image_path = os.path.join(images_directory, file_name)
                outputs = predictor.predict(image_path, fuse_model=False)
                parsed_outputs = self.parse_model_outputs(outputs)
                parsed_outputs_by_patch[file_name] = parsed_outputs
        
        return parsed_outputs_by_patch
    
    
    def predict_and_parse_dataset(self, images_directory, predictor):
        """
        Process all images in the dataset, predict on their patches, and collect outputs in the format of the parsed model output.
        Args:
            images_directory (str): The directory path where image patches are stored.
            predictor (callable): The predictor object to use for making predictions.
        Returns:
            dict: A dictionary containing the image base names as keys and prediction outputs for each image as values.
        """
        image_numbers = self.get_image_numbers(images_directory)
        parsed_outputs_by_image = {}

        for image_number in image_numbers:
            parsed_outputs_by_patch = self.predict_and_parse_image_patches(image_number, images_directory, predictor)
            parsed_outputs_by_image[image_number] = parsed_outputs_by_patch

        return parsed_outputs_by_image
