from src.segmentation.evaluation.base_evaluator import BaseEvaluator

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
