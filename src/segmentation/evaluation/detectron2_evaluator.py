from src.segmentation.evaluation.base_evaluator import BaseEvaluator
import json
import copy

class Detectron2Evaluator(BaseEvaluator):
    def __init__(self, num_classes, coco_annotations_file_path):
        super().__init__(num_classes)
        self.coco_data = self.load_coco_annotations(coco_annotations_file_path)
        
    @staticmethod
    def load_coco_annotations(file_path):
        with open(file_path, 'r') as f:
            coco_data = json.load(f)
        return coco_data

    def get_annotations_for_image(self, file_name):
        # Find the image entry based on the file name
        image_entry = next((img for img in self.coco_data['images'] if img['file_name'] == file_name), None)
        if not image_entry:
            raise ValueError(f"No image with file name {file_name} found in annotations.")
        
        image_id = image_entry['id']
        
        # Get all annotations for this image
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]
        
        return image_entry, annotations

    def parse_annotations(self, file_path):
        """ Parse the COCO annotations and return the ground truth bounding boxes and class IDs."""
        
        _, annotations = self.get_annotations_for_image(file_path)
        ground_truths = []
        for annotation in annotations:
            bbox = annotation['bbox']  # COCO provides bboxes as [x_min, y_min, width, height]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            class_id = annotation['category_id'] - 1  # COCO class IDs are 1-indexed
            ground_truths.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'class_id': class_id
            })
        return ground_truths

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
    