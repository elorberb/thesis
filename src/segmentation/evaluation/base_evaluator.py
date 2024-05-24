from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from PIL import Image
import copy
import os
import re

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
    
    
    @ staticmethod
    def normalize_confusion_matrix(confusion_matrix):
        """
        Normalize a confusion matrix to show proportions instead of counts.

        Args:
            confusion_matrix (numpy.ndarray): The confusion matrix to normalize.

        Returns:
            numpy.ndarray: The normalized confusion matrix.
        """
        normalized_matrix = confusion_matrix.astype(np.float32)
        for i in range(len(normalized_matrix)):
            row_sum = normalized_matrix[i, :].sum()
            if row_sum > 0:
                normalized_matrix[i, :] /= row_sum
        return normalized_matrix
    
    
    @staticmethod
    def get_image_numbers(images_directory):
        """
        Extract unique base image numbers from a directory containing image patches.

        Args:
            images_directory (str): The directory path where image patches are stored.

        Returns:
            set: A set containing unique base image numbers.
        """
        image_numbers = set()
        pattern = re.compile(r"IMG_\d+")

        # Iterate over each file in the directory
        for file_name in os.listdir(images_directory):
            match = pattern.search(file_name)
            if match:
                image_number = match.group(0)  # Extracts the matched part of the filename, e.g., IMG_0019
                image_numbers.add(image_number)

        return image_numbers

    
    def compute_confusion_matrix(self, matches, misclassifications, false_positives, false_negatives, gt_boxes, pred_boxes, normalize=False, single_class=False):
        num_classes = self.num_classes  # Assuming this is set correctly
        cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)  # Last row/col for FPs/FNs

        # True Positives
        for gt_idx, pred_idx, _ in matches:
            class_id = gt_boxes[gt_idx]['class_id']
            cm[class_id, class_id] += 1

        # Misclassifications
        for gt_idx, pred_idx, _ in misclassifications:
            true_class = gt_boxes[gt_idx]['class_id']
            pred_class = pred_boxes[pred_idx]['class_id']
            cm[true_class, pred_class] += 1

        # False Positives
        for pred in false_positives:
            pred_class = pred['class_id']
            cm[num_classes, pred_class] += 1

        # False Negatives
        for fn in false_negatives:
            true_class = fn['class_id']
            cm[true_class, num_classes] += 1

        if normalize:
            cm = cm.astype(np.float32)
            for i in range(len(cm)):
                if cm[i, :].sum() > 0:
                    cm[i, :] /= cm[i, :].sum()

        return cm


    def match_predictions(self, gt_boxes, pred_boxes, iou_thresh=0.5):
        matches = []
        misclassifications = []
        false_positives = []
        detected = set()

        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_match = None

            for gt_idx, gt in enumerate(gt_boxes):
                current_iou = self.iou(pred['bbox'], gt['bbox'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match = (gt_idx, pred_idx, current_iou)

            if best_match and best_iou >= iou_thresh:
                gt_idx, pred_idx, _ = best_match
                if gt_idx not in detected:  # Ensure the gt has not been matched already
                    if gt_boxes[gt_idx]['class_id'] == pred['class_id']:
                        matches.append(best_match)
                        detected.add(gt_idx)
                    else:
                        misclassifications.append(best_match)
                        detected.add(gt_idx)
                # If gt_idx was already used, you can decide to handle differently or ignore
            else:
                false_positives.append(pred)  # No good match found, treat as false positive

        # Identify false negatives
        false_negatives = [{'bbox': gt['bbox'], 'class_id': gt['class_id']} for gt_idx, gt in enumerate(gt_boxes) if gt_idx not in detected]

        return matches, misclassifications, false_positives, false_negatives


    def calculate_metrics(self, confusion_matrix, single_class=False):
        """
        Calculate precision and recall for each class and overall based on the provided confusion matrix.
        Args:
            confusion_matrix (np.array): The confusion matrix, with shape (num_classes+1, num_classes+1).
        Returns:
            dict: A dictionary containing metrics such as class-wise precision and recall, and overall metrics.
        """
        if single_class:
            # This block assumes confusion_matrix is a 2x2 matrix for a single class.
            tp = confusion_matrix[0, 0]
            fp = confusion_matrix[0, 1] + confusion_matrix[1, 0]
            fn = confusion_matrix[1, 0]
            precision = self.calculate_precision(tp, fp)
            recall = self.calculate_recall(tp, fn)
            return {"precision": precision, "recall": recall}
        
        num_classes = confusion_matrix.shape[0] - 1
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        
        # Compute precision and recall for each class
        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            
            # Correctly compute fp by summing the misclassifications for the predicted class
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp

            precision[i] = self.calculate_precision(tp, fp)
            recall[i] = self.calculate_recall(tp, fn)
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        overall_precision = self.calculate_precision(overall_tp, overall_fp)
        overall_recall = self.calculate_recall(overall_tp, overall_fn)

        return {
            "precision": overall_precision,
            "recall": overall_recall,
            "class_wise_precision": precision,
            "class_wise_recall": recall
        }



    def evaluate_patch(self, gt_boxes, pred_boxes, iou_thresh=0.5, single_class=False):
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

        matches, misclassifications, false_positives, false_negatives = self.match_predictions(working_gt_boxes, working_pred_boxes, iou_thresh)
        confusion_matrix = self.compute_confusion_matrix(matches, misclassifications, false_positives, false_negatives, working_gt_boxes, working_pred_boxes, single_class=single_class)
        normalized_confusion_matrix = self.compute_confusion_matrix(matches, misclassifications, false_positives, false_negatives, working_gt_boxes, working_pred_boxes, normalize=True, single_class=single_class)
        metrics = self.calculate_metrics(confusion_matrix, single_class=single_class)

        return {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix,
            "normalized_confusion_matrix": normalized_confusion_matrix
        }
        

    def evaluate_image(self, patches_gt_boxes_dict, patches_pred_boxes_dict, iou_thresh=0.5, single_class=False):
        """
        Evaluate all patches of a single image and aggregate the results, with input as dictionaries.

        Args:
            patches_gt_boxes_dict (dict): A dictionary where each key is a patch identifier and the value is a list of ground truth boxes for that patch.
            patches_pred_boxes_dict (dict): A dictionary where each key is a patch identifier and the value is a list of predicted boxes for that patch.
            iou_thresh (float): The IoU threshold for determining matches.
            single_class (bool): Whether to treat all detections as a single class.

        Returns:
            dict: A dictionary containing metrics, confusion matrix, and normalized confusion matrix for the entire image.
        """
        num_classes = self.num_classes
        global_conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

        # Evaluate each patch using evaluate_patch function and aggregate the results
        for patch_key in patches_gt_boxes_dict.keys():
            patch_result = self.evaluate_patch(patches_gt_boxes_dict[patch_key], patches_pred_boxes_dict[patch_key], iou_thresh, single_class)
            global_conf_matrix += patch_result['confusion_matrix']

        # Normalize the global confusion matrix
        normalized_conf_matrix = self.normalize_confusion_matrix(global_conf_matrix)

        # Calculate metrics based on the global confusion matrix
        metrics = self.calculate_metrics(global_conf_matrix, single_class=single_class)

        return {
            "metrics": metrics,
            "confusion_matrix": global_conf_matrix,
            "normalized_confusion_matrix": normalized_conf_matrix
        }
        
        
    def evaluate_dataset(self, dataset_gt_boxes_dict, dataset_pred_boxes_dict, iou_thresh=0.5, single_class=False):
        """
        Evaluate all images in the dataset and aggregate the results, with input as dictionaries of dictionaries.

        Args:
            dataset_gt_boxes_dict (dict): A dictionary where each key is an image number and the value is 
                                        a dictionary of ground truth boxes for each patch of that image.
            dataset_pred_boxes_dict (dict): A dictionary where each key is an image number and the value is 
                                            a dictionary of predicted boxes for each patch of that image.
            iou_thresh (float): The IoU threshold for determining matches.
            single_class (bool): Whether to treat all detections as a single class.

        Returns:
            dict: A dictionary containing aggregated metrics, confusion matrix, and normalized confusion matrix for the entire dataset.
        """
        num_classes = self.num_classes
        dataset_conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

        # Iterate over each image in the dataset
        for image_number in dataset_gt_boxes_dict.keys():
            patches_gt_boxes_dict = dataset_gt_boxes_dict[image_number]
            patches_pred_boxes_dict = dataset_pred_boxes_dict[image_number]

            # Evaluate each image using the evaluate_image function
            image_results = self.evaluate_image(patches_gt_boxes_dict, patches_pred_boxes_dict, iou_thresh, single_class)

            # Aggregate the confusion matrices
            dataset_conf_matrix += image_results['confusion_matrix']

        # normalize the dataset confusion matrix
        normalized_conf_matrix = self.normalize_confusion_matrix(dataset_conf_matrix)

        # Calculate metrics based on the aggregated dataset confusion matrix
        metrics = self.calculate_metrics(dataset_conf_matrix)

        return {
            "metrics": metrics,
            "confusion_matrix": dataset_conf_matrix,
            "normalized_confusion_matrix": normalized_conf_matrix
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
    def plot_boxes(image, boxes, class_label=None, is_ground_truth=True):
        """
        Plots bounding boxes on the image for a specific class label.

        Args:
            image (PIL Image or str): The image on which to plot, or the path to the image.
            boxes (list): List of dictionaries containing bounding boxes with 'bbox' and 'class_id'.
            class_label (int, optional): Specific class label to plot. If None, plots all classes.
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
            if class_label is None or box['class_id'] == class_label:
                x_min, y_min, x_max, y_max = box['bbox']
                cls_id = box['class_id']
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=colors[cls_id], facecolor='none', linestyle=linestyle, label=class_labels[cls_id])
                ax.add_patch(rect)

        # Set legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Removing duplicates in legend
        ax.legend(by_label.values(), by_label.keys())

        plt.show()