from src.segmentation.evaluation.base_evaluator import BaseEvaluator
import re
import os

class UltralyticsEvaluator(BaseEvaluator):
    def __init__(self, num_classes, image_size):
        super().__init__(num_classes)
        self.image_size = image_size


    def parse_annotations(self, file_path):
        """ Parse the annotations from a file and return the ground truth bounding boxes and class IDs."""
        ground_truths = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * self.image_size
                y_center = float(parts[2]) * self.image_size
                width = float(parts[3]) * self.image_size
                height = float(parts[4]) * self.image_size
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                ground_truths.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'class_id': class_id})
        return ground_truths
    
    
    def get_annotations_for_image_patches(self, image_number, annotations_directory):
        """
        Retrieve all patch annotations for a given base image name and organize them in a dictionary.
        Args:
            image_number (str): The base name of the image, e.g., 'IMG_2157'.
            annotations_directory (str): The directory path where annotation files are stored.
        Returns:
            dict: A dictionary where each key is the patch's file name and the value is a list of dictionaries,
                each representing bounding boxes and class IDs for that patch.
        """
        patches_gt_boxes = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.txt$")  # Adjusted to match .txt files for Ultralytics

        for file_name in os.listdir(annotations_directory):
            if pattern.match(file_name):
                file_path = os.path.join(annotations_directory, file_name)
                gt_boxes = self.parse_annotations(file_path)
                file_name = file_name.replace('.txt', '.png')  # Adjusted to match .jpg files for Ultralytics
                patches_gt_boxes[file_name] = gt_boxes
        
        return patches_gt_boxes
    

    def get_annotations_for_dataset(self, annotations_directory):
        """
        Collect annotations for all images in the dataset from their respective patches.
        Args:
            annotations_directory (str): The directory path where annotation files are stored.
        Returns:
            dict: A dictionary containing the image base names as keys and ground truth for each image as values.
        """
        image_gt_boxes = {}
        image_numbers = set()

        # Extract base image numbers from filenames
        for file_name in os.listdir(annotations_directory):
            match = re.match(r"(IMG_\d+)_p\d+\.txt", file_name)
            if match:
                image_numbers.add(match.group(1))

        for image_number in image_numbers:
            patches_gt_boxes = self.get_annotations_for_image_patches(image_number, annotations_directory)
            image_gt_boxes[image_number] = patches_gt_boxes

        return image_gt_boxes



    def parse_model_outputs(self, outputs):
        """ Parse the model outputs to extract bounding boxes, confidence scores, and class IDs."""
        parsed_outputs = []
        for detection in outputs[0].boxes.data:
            x_min, y_min, x_max, y_max, confidence, class_id = detection.cpu().numpy()
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
                image_path = os.path.join(images_directory, file_name)
                outputs = predictor(image_path)
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
