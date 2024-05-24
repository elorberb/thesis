import pytest
import numpy as np
from src.segmentation.evaluation.base_evaluator import BaseEvaluator

class ConcreteEvaluator(BaseEvaluator):
    def parse_annotations(self, file_path):
        pass

    def parse_model_outputs(self, outputs):
        pass

@pytest.fixture
def evaluator():
    return ConcreteEvaluator()

def test_calculate_precision(evaluator):
    assert evaluator.calculate_precision(true_positives=5, false_positives=3) == 5 / 8
    assert evaluator.calculate_precision(true_positives=0, false_positives=0) == 0

def test_calculate_recall(evaluator):
    assert evaluator.calculate_recall(true_positives=5, false_negatives=3) == 5 / 8
    assert evaluator.calculate_recall(true_positives=0, false_negatives=0) == 0

def test_normalize_confusion_matrix(evaluator):
    cm = np.array([[5, 3], [2, 4]])
    normalized_cm = evaluator.normalize_confusion_matrix(cm)
    expected_normalized_cm = np.array([[5 / 8, 3 / 8], [2 / 6, 4 / 6]])
    np.testing.assert_array_almost_equal(normalized_cm, expected_normalized_cm)

def test_get_image_numbers(evaluator, mocker):
    mocker.patch('os.listdir', return_value=['IMG_0019_patch_1.png', 'IMG_0020_patch_1.png', 'some_other_file.txt'])
    image_numbers = evaluator.get_image_numbers('/dummy/path')
    assert image_numbers == {'IMG_0019', 'IMG_0020'}

def test_compute_confusion_matrix(evaluator):
    
    gt_boxes = [
        {'bbox': [2, 2, 4, 4], 'class_id': 0}, # matches
        {'bbox': [4, 4, 6, 6], 'class_id': 1}, # misclassifications
        {'bbox': [10, 10, 12, 12], 'class_id': 2} # false negative
    ]
    pred_boxes = [
        {'bbox': [2, 2, 4, 4], 'class_id': 0}, # matches
        {'bbox': [4, 4, 6, 6],'class_id': 2}, # misclassifications
        {'bbox': [12, 12, 18, 18], 'class_id': 0} # false positive
    ]
    # results from running match_predictions
    matches = [(0, 0, 1)]
    misclassifications = [(1, 1, 1)]
    false_positives = [{'bbox': [12, 12, 18, 18], 'class_id': 0}]
    false_negatives = [{'bbox': [10, 10, 12, 12], 'class_id': 2}]
    
    cm = evaluator.compute_confusion_matrix(matches, misclassifications, false_positives, false_negatives, gt_boxes, pred_boxes)

    expected_cm = np.array([
        [1, 0, 0, 0],  # Class 0: 1 TP, 1 FN
        [0, 0, 1, 0],  # Class 1: 1 misclassification - pred class 2 instead of 1 (FP)
        [0, 0, 0, 1],  # Class 2: 1 misclassification - missed detection of class 2 (FN)
        [1, 0, 0, 0]   # False Positives - predicted class 0 when it is background
    ])

    np.testing.assert_array_equal(cm, expected_cm)

def test_match_predictions(evaluator):
    gt_boxes = [
        {'bbox': [2, 2, 4, 4], 'class_id': 0}, # matches
        {'bbox': [4, 4, 6, 6], 'class_id': 1}, # misclassifications
        {'bbox': [10, 10, 12, 12], 'class_id': 2} # false negative
    ]
    pred_boxes = [
        {'bbox': [2, 2, 4, 4], 'class_id': 0}, # matches
        {'bbox': [4, 4, 6, 6],'class_id': 2}, # misclassifications
        {'bbox': [12, 12, 18, 18], 'class_id': 0} # false positive
    ]
    
    matches, misclassifications, false_positives, false_negatives = evaluator.match_predictions(gt_boxes, pred_boxes, 0.5)
    print(matches, misclassifications, false_positives, false_negatives)
    # Validate matches
    assert len(matches) == 1
    assert matches[0] == (0, 0, 1)  # Example match

    # Validate misclassifications
    assert len(misclassifications) == 1
    assert misclassifications[0] == (1, 1, 1)  # Example misclassification

    # Validate false positives
    assert len(false_positives) == 1
    assert false_positives[0] == pred_boxes[2]

    # Validate false negatives
    assert len(false_negatives) == 1
    assert false_negatives[0] == gt_boxes[2]

def test_calculate_metrics(evaluator):
    cm = np.array([
        [3, 1, 0, 2],  # True 0: 3 TP, 1 FP (Pred as 1), 0 FP (Pred as 2), 2 FN
        [1, 4, 2, 1],  # True 1: 1 FP (Pred as 0), 4 TP, 2 FP (Pred as 2), 1 FN
        [0, 2, 5, 1],  # True 2: 0 FP (Pred as 0), 2 FP (Pred as 1), 5 TP, 1 FN
        [1, 1, 1, 0]   # FP: 1 FP for each class
    ])
    metrics = evaluator.calculate_metrics(cm)
    
    expected_class_wise_precision = [0.6, 0.5, 0.625]  
    expected_class_wise_recall = [0.5, 0.5, 0.625]

    assert np.round(metrics['precision'], 3) == pytest.approx(0.571)
    assert np.round(metrics['recall'], 3) == pytest.approx(0.545)
    np.testing.assert_array_almost_equal(metrics['class_wise_precision'], expected_class_wise_precision)
    np.testing.assert_array_almost_equal(metrics['class_wise_recall'], expected_class_wise_recall)