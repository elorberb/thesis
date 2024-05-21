from src.segmentation.evaluation.base_evaluator import BaseEvaluator
import json
import os
import cv2
import re


class Detectron2Evaluator(BaseEvaluator):
    def __init__(self, num_classes, coco_annotations_file_path):
        super().__init__(num_classes)
        self.coco_data = self.load_coco_annotations(coco_annotations_file_path)
        
    @staticmethod
    def load_coco_annotations(file_path):
        with open(file_path, 'r') as f:
            coco_data = json.load(f)
        return coco_data
    

    def get_annotations_for_patch(self, file_name):
        # Find the patch image entry based on the file name
        image_entry = next((img for img in self.coco_data['images'] if img['file_name'] == file_name), None)
        if not image_entry:
            raise ValueError(f"No image with file name {file_name} found in annotations.")
        
        image_id = image_entry['id']
        
        # Get all annotations for this image
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        
        return image_entry, annotations
    
    
    def parse_annotations(self, file_path):
        """ Parse the COCO annotations and return the ground truth bounding boxes and class IDs."""
        
        _, annotations = self.get_annotations_for_patch(file_path)
        ground_truths = []
        for annotation in annotations:
            bbox = annotation['bbox']  # COCO provides bboxes as [x_min, y_min, width, height]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            class_id = annotation['category_id']  
            ground_truths.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'class_id': class_id - 1 # COCO class IDs are 1-indexed
            })
        return ground_truths
    
    
    def get_annotations_for_image_patches(self, image_number):
        """
        Retrieve all patch annotations for a given base image name and organize them in a dictionary.
        
        Args:
            image_number (str): The base name of the image, e.g., 'IMG_2157'.
        
        Returns:
            dict: A dictionary where each key is the patch's file name and the value is a list of dictionaries,
                each representing bounding boxes and class IDs for that patch.
        """
        patches_gt_boxes = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.\w+$")  # Regex pattern to match specific file names

        # Iterate through all images in the COCO dataset to find matches
        for img in self.coco_data['images']:
            if pattern.match(img['file_name']):  # Check if the base image name is in the file name
                gt_boxes = self.parse_annotations(img['file_name'])
                patches_gt_boxes[img['file_name']] = gt_boxes
        
        return patches_gt_boxes
    
    
    def get_annotations_for_dataset(self, images_directory):
        """
        Collect annotations for all images in the dataset from their respective patches.
        
        Args:
            images_directory (str): The directory path where image patches are stored.
        
        Returns:
            dict: A dictionary containing the image base names as keys and ground truth for each image as values.
        """
        image_numbers = self.get_image_numbers(images_directory)
                
        image_gt_boxes = {}
        for image_number in image_numbers:
                patches_gt_boxes = self.get_annotations_for_image_patches(image_number)
                image_gt_boxes[image_number] = patches_gt_boxes
        
        return image_gt_boxes



    def parse_model_outputs(self, outputs):
        """ Parse the Detectron2 model outputs to extract bounding boxes, confidence scores, and class IDs."""
        parsed_outputs = []
        instances = outputs['instances']
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()
        
        for bbox, score, class_id in zip(pred_boxes, scores, pred_classes):
            x_min, y_min, x_max, y_max = bbox
            parsed_outputs.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'score': score,
                'class_id': class_id - 1 # Detectron2 class IDs are 1-indexed
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
        for file_name in os.listdir(images_directory):
            if image_number in file_name and 'label_ground-truth' not in file_name:
                
                # Load the image
                image_path = os.path.join(images_directory, file_name)

                # Predict using the Detectron2 model
                outputs = predictor(cv2.imread(image_path))

                # Parse the model outputs
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
        # get the image numbers from the images directory
        image_numbers = self.get_image_numbers(images_directory)
                
        parsed_outputs_by_image = {}
        for image_number in image_numbers:
                parsed_outputs_by_patch = self.predict_and_parse_image_patches(image_number, images_directory, predictor)
                parsed_outputs_by_image[image_number] = parsed_outputs_by_patch
        
        return parsed_outputs_by_image
    