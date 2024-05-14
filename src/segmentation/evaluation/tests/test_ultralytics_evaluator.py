import unittest
from src.segmentation.evaluation.ultralytics_evaluator import UltralyticsEvaluator
import numpy as np

class TestUltralyticsEvaluator(unittest.TestCase):
    def setUp(self):
        # Setup that runs before each test method
        self.evaluator = UltralyticsEvaluator(num_classes=3, image_size=512)

    def test_iou_no_overlap(self):
        boxA = [1, 1, 2, 2]
        boxB = [3, 3, 4, 4]
        self.assertEqual(self.evaluator.iou(boxA, boxB), 0)

    def test_iou_complete_overlap(self):
        boxA = [1, 1, 3, 3]
        boxB = [1, 1, 3, 3]
        self.assertEqual(self.evaluator.iou(boxA, boxB), 1)

    def test_calculate_metrics(self):
        matches = [(0, 0, 0.9)]
        detected = [0]
        gt_boxes = [{'bbox': [1, 1, 3, 3], 'class_id': 1}]
        pred_boxes = [{'bbox': [1, 1, 3, 3], 'class_id': 1}, {'bbox': [4, 4, 6, 6], 'class_id': 2}]
        metrics = self.evaluator.calculate_metrics(matches, detected, gt_boxes, pred_boxes)
        expected_metrics = {'precision': 0.5, 'recall': 1.0}
        self.assertEqual(metrics, expected_metrics)
        
    def test_calculate_precision(self):
        self.assertEqual(self.evaluator.calculate_precision(10, 90), 0.1)
        self.assertEqual(self.evaluator.calculate_precision(0, 0), 0)  # Edge case

    def test_calculate_recall(self):
        self.assertEqual(np.round(self.evaluator.calculate_recall(10, 20)), np.round(0.3333))
        self.assertEqual(self.evaluator.calculate_recall(0, 0), 0)  # Edge case

    def test_compute_confusion_matrix(self):
        matches = [(0, 0, 0.9), (1, 1, 0.8)]
        false_positives = [{'class_id': 1}, {'class_id': 2}]
        false_negatives = [{'class_id': 1}]
        gt_boxes = [{'class_id': 1}, {'class_id': 1}]
        pred_boxes = [{'class_id': 1}, {'class_id': 1}]
        cm = self.evaluator.compute_confusion_matrix(matches, false_positives, false_negatives, gt_boxes, pred_boxes)
        expected_matrix = np.zeros((4, 4))  # Assuming num_classes=3 + background
        expected_matrix[1, 1] = 2
        expected_matrix[-1, 1] = 1
        expected_matrix[-1, 2] = 1
        expected_matrix[1, -1] = 1
        np.testing.assert_array_equal(cm, expected_matrix)
        
    def test_match_predictions(self):
        gt_boxes = [{'bbox': [1, 1, 3, 3], 'class_id': 1}, {'bbox': [5, 5, 6, 6], 'class_id': 2}]
        pred_boxes = [{'bbox': [1, 1, 3, 3], 'class_id': 1, 'score': 0.9}, {'bbox': [5, 5, 6, 6], 'class_id': 2, 'score': 0.95}]
        matches, fps, fns = self.evaluator.match_predictions(gt_boxes, pred_boxes, iou_thresh=0.5)
        self.assertEqual(len(matches), 2)  # both should match
        self.assertEqual(len(fps), 0)
        self.assertEqual(len(fns), 0)

        # Test case with no overlap
        pred_boxes = [{'bbox': [10, 10, 12, 12], 'class_id': 1, 'score': 0.8}]
        matches, fps, fns = self.evaluator.match_predictions(gt_boxes, pred_boxes, iou_thresh=0.5)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(fps), 1)
        self.assertEqual(len(fns), 2)  # both GTs unmatched


if __name__ == '__main__':
    unittest.main()
