from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from PIL import Image
import copy

class BaseEvaluator(ABC):
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    @abstractmethod
    def parse_annotations(self, file_path):
        """ Abstract method to parse the annotations for any framework."""
        pass

    @abstractmethod
    def parse_model_outputs(self, outputs):
        """ Abstract method to parse the model outputs for any framework."""
        pass

    @staticmethod
    def iou(boxA, boxB):
        """ Calculate the Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @staticmethod
    def calculate_precision(true_positives, false_positives):
        """Calculate precision from true positives and false positives."""
        if true_positives + false_positives == 0:
            return 0
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def calculate_recall(true_positives, false_negatives):
        """Calculate recall from true positives and false negatives."""
        if true_positives + false_negatives == 0:
            return 0
        return true_positives / (true_positives + false_negatives)

    def compute_confusion_matrix(self, matches, false_positives, false_negatives, gt_boxes, pred_boxes, normalize=False, single_class=False):
        """Compute confusion matrix based on matches and misclassifications."""
        if single_class:
            cm = np.zeros((2, 2), dtype=int)
        else:
            cm = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=int)

        for match in matches:
            gt_idx, pred_idx, _ = match
            gt_class = 0 if single_class else gt_boxes[gt_idx]['class_id']
            pred_class = 0 if single_class else pred_boxes[pred_idx]['class_id']
            cm[gt_class, pred_class] += 1

        for fp in false_positives:
            pred_class = 0 if single_class else fp['class_id']
            cm[-1, pred_class] += 1

        for fn in false_negatives:
            gt_class = 0 if single_class else fn['class_id']
            cm[gt_class, -1] += 1

        if normalize:
            cm = cm.astype(np.float32)
            for i in range(len(cm) - 1):
                if cm[i, :].sum() > 0:
                    cm[i, :] /= cm[i, :].sum()
            if cm[-1, :-1].sum() > 0:
                cm[-1, :-1] /= cm[-1, :-1].sum()

        return cm

    def match_predictions(self, gt_boxes, pred_boxes, iou_thresh=0.5):
        """Match predictions to ground truth boxes based on IoU threshold, and classify predictions."""
        matches = []
        detected = []
        false_positives = []
        unmatched_gts = set(range(len(gt_boxes)))

        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_match = None

            for gt_idx, gt in enumerate(gt_boxes):
                if gt['class_id'] == pred['class_id']:
                    current_iou = self.iou(pred['bbox'], gt['bbox'])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_match = (gt_idx, pred_idx, best_iou)

            if best_match and best_iou >= iou_thresh:
                if best_match[0] not in detected:
                    matches.append(best_match)
                    detected.append(best_match[0])
                    unmatched_gts.discard(best_match[0])
                else:
                    false_positives.append(pred)
            else:
                false_positives.append(pred)

        false_negatives = [{'bbox': gt['bbox'], 'class_id': gt['class_id']} for gt_idx, gt in enumerate(gt_boxes) if gt_idx in unmatched_gts]

        return matches, false_positives, false_negatives

    def calculate_metrics(self, confusion_matrix, single_class=False):
        """Calculate precision and recall based on the confusion matrix."""
        if single_class:
            tp = confusion_matrix[0, 0]
            fp = confusion_matrix[-1, 0]
            fn = confusion_matrix[0, -1]
            precision = self.calculate_precision(tp, fp)
            recall = self.calculate_recall(tp, fn)
            return {"precision": precision, "recall": recall}

        num_classes = confusion_matrix.shape[0] - 1
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        overall_true_positives = np.diag(confusion_matrix)[:num_classes].sum()
        overall_false_positives = confusion_matrix[-1, :num_classes].sum()
        overall_false_negatives = confusion_matrix[:num_classes, -1].sum()

        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[-1, i]
            fn = confusion_matrix[i, -1]

            precision[i] = self.calculate_precision(tp, fp)
            recall[i] = self.calculate_recall(tp, fn)

        overall_precision = self.calculate_precision(overall_true_positives, overall_false_positives)
        overall_recall = self.calculate_recall(overall_true_positives, overall_false_negatives)

        return {
            "precision": overall_precision,
            "recall": overall_recall,
            "class_wise_precision": precision,
            "class_wise_recall": recall
        }

    def evaluate(self, gt_boxes, pred_boxes, iou_thresh=0.5, single_class=False):
        """ Evaluate the model outputs against the ground truth annotations.
            If single_class is True, convert all classes to a single class.
        """
        working_gt_boxes = copy.deepcopy(gt_boxes)
        working_pred_boxes = copy.deepcopy(pred_boxes)
        
        if single_class:
            for box in working_gt_boxes:
                box['class_id'] = 0
            for box in working_pred_boxes:
                box['class_id'] = 0

        matches, false_positives, false_negatives = self.match_predictions(working_gt_boxes, working_pred_boxes, iou_thresh)
        confusion_matrix = self.compute_confusion_matrix(matches, false_positives, false_negatives, working_gt_boxes, working_pred_boxes, single_class=single_class)
        normalized_confusion_matrix = self.compute_confusion_matrix(matches, false_positives, false_negatives, working_gt_boxes, working_pred_boxes, normalize=True, single_class=single_class)
        metrics = self.calculate_metrics(confusion_matrix, single_class=single_class)

        return {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix,
            "normalized_confusion_matrix": normalized_confusion_matrix
        }

    # ------------- Plot functions ------------- 
    @staticmethod
    def plot_confusion_matrix(cm, class_names=['Clear', 'Cloudy', 'Amber']):
        """Plot the confusion matrix."""
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="0.2f" if cm.dtype == np.float32 else "d", cmap="Blues", xticklabels=class_names + ['Background'], yticklabels=class_names + ['Background'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    @staticmethod
    def plot_boxes(image, boxes, is_ground_truth=True):
        """
        Plots bounding boxes on the image.

        Args:
            image (PIL Image or str): The image on which to plot, or the path to the image.
            boxes (list): List of dictionaries containing bounding boxes with 'bbox' and 'class_id'.
            is_ground_truth (bool): Flag indicating if the boxes are ground truth. Defaults to True.
        """
        # If image is the path to the image, read it
        if isinstance(image, str):
            image = Image.open(image)

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Set class labels (assuming 3 classes as described: clear, cloudy, amber)
        class_labels = {0: 'clear', 1: 'cloudy', 2: 'amber'}
        
        # Define colors for different classes
        gt_colors = {0: 'darkgrey', 1: 'lightgrey', 2: 'goldenrod'}
        pred_colors = {0: 'grey', 1: 'white', 2: 'orange'}

        # Select appropriate colors
        colors = gt_colors if is_ground_truth else pred_colors
        linestyle = 'dashed' if is_ground_truth else 'solid'

        # Plot boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = box['bbox']
            cls_id = box['class_id']
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=colors[cls_id], facecolor='none', linestyle=linestyle, label=class_labels[cls_id])
            ax.add_patch(rect)

        # Set legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Removing duplicates in legend
        ax.legend(by_label.values(), by_label.keys())

        plt.show()