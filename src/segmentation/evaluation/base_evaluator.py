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
        """Abstract method to parse the annotations for any framework."""
        pass

    @abstractmethod
    def parse_model_outputs(self, outputs):
        """Abstract method to parse the model outputs for any framework."""
        pass

    @staticmethod
    def iou(boxA, boxB):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
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

    @staticmethod
    def calculate_f1_score(precision, recall):
        """
        Calculate the F1 score given precision and recall.
        Returns 0 if both precision and recall are zero.
        """
        if (precision + recall) == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
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
                image_number = match.group(
                    0
                )  # Extracts the matched part of the filename, e.g., IMG_0019
                image_numbers.add(image_number)

        return image_numbers

    def compute_confusion_matrix(
        self,
        matches,
        misclassifications,
        false_positives,
        false_negatives,
        gt_boxes,
        pred_boxes,
        normalize=False,
        single_class=False,
    ):
        if single_class:
            num_classes = 1  # Use 1 to represent the single class scenario
        else:
            num_classes = self.num_classes  # Use the actual number of classes

        # Initialize the confusion matrix
        cm = np.zeros(
            (num_classes + 1, num_classes + 1), dtype=int
        )  # Last row/col for FPs/FNs

        # True Positives
        for gt_idx, pred_idx, _ in matches:
            gt_class = 0 if single_class else gt_boxes[gt_idx]["class_id"]
            pred_class = 0 if single_class else pred_boxes[pred_idx]["class_id"]
            cm[gt_class, pred_class] += 1

        # Misclassifications
        for gt_idx, pred_idx, _ in misclassifications:
            true_class = 0 if single_class else gt_boxes[gt_idx]["class_id"]
            pred_class = 0 if single_class else pred_boxes[pred_idx]["class_id"]
            cm[true_class, pred_class] += 1

        # False Positives
        for pred in false_positives:
            pred_class = 0 if single_class else pred["class_id"]
            cm[num_classes, pred_class] += 1

        # False Negatives
        for fn in false_negatives:
            true_class = 0 if single_class else fn["class_id"]
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
                current_iou = self.iou(pred["bbox"], gt["bbox"])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match = (gt_idx, pred_idx, current_iou)

            if best_match and best_iou >= iou_thresh:
                gt_idx, pred_idx, _ = best_match
                if gt_idx not in detected:  # Ensure the gt has not been matched already
                    if gt_boxes[gt_idx]["class_id"] == pred["class_id"]:
                        matches.append(best_match)
                        detected.add(gt_idx)
                    else:
                        misclassifications.append(best_match)
                        detected.add(gt_idx)
                # If gt_idx was already used, you can decide to handle differently or ignore
            else:
                false_positives.append(
                    pred
                )  # No good match found, treat as false positive

        # Identify false negatives
        false_negatives = [
            {"bbox": gt["bbox"], "class_id": gt["class_id"]}
            for gt_idx, gt in enumerate(gt_boxes)
            if gt_idx not in detected
        ]

        return matches, misclassifications, false_positives, false_negatives

    def calculate_metrics(
        self,
        confusion_matrix,
        single_class=False,
    ):
        """
        Calculate precision, recall, and F1 score for each class and overall based on the confusion matrix.
        Optionally, if dataset-level ground truth and predictions (with scores) are provided,
        also compute mAP using the COCO protocol.

        Args:
            confusion_matrix (np.array): Confusion matrix of shape (num_classes+1, num_classes+1).
            single_class (bool): Flag for single-class evaluation.
            dataset_gt_boxes (dict, optional): Mapping from image_id to list of GT boxes.
            dataset_pred_boxes (dict, optional): Mapping from image_id to list of predicted boxes.
            class_ids (list, optional): List of class IDs to compute mAP for.
            iou_thresholds (iterable): IoU thresholds over which to compute AP (default: 0.5 to 0.95).

        Returns:
            dict: Dictionary with overall and class-wise precision, recall, F1, and optionally mAP.
        """
        if single_class:
            tp = confusion_matrix[0, 0]
            fp = confusion_matrix[0, 1]
            fn = confusion_matrix[1, 0]
            precision = self.calculate_precision(tp, fp)
            recall = self.calculate_recall(tp, fn)
            f1 = self.calculate_f1_score(precision=precision, recall=recall)
            metrics_dict = {"precision": precision, "recall": recall, "f1": f1}
        else:
            num_classes = confusion_matrix.shape[0] - 1
            precision_arr = np.zeros(num_classes)
            recall_arr = np.zeros(num_classes)
            f1_arr = np.zeros(num_classes)
            overall_tp = overall_fp = overall_fn = 0

            for i in range(num_classes):
                tp = confusion_matrix[i, i]
                fp = np.sum(confusion_matrix[:, i]) - tp
                fn = np.sum(confusion_matrix[i, :]) - tp

                precision_arr[i] = self.calculate_precision(tp, fp)
                recall_arr[i] = self.calculate_recall(tp, fn)
                f1_arr[i] = self.calculate_f1_score(precision_arr[i], recall_arr[i])
                overall_tp += tp
                overall_fp += fp
                overall_fn += fn

            overall_precision = self.calculate_precision(overall_tp, overall_fp)
            overall_recall = self.calculate_recall(overall_tp, overall_fn)
            overall_f1 = self.calculate_f1_score(overall_precision, overall_recall)
            metrics_dict = {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "class_wise_precision": precision_arr,
                "class_wise_recall": recall_arr,
                "class_wise_f1": f1_arr,
            }

        return metrics_dict

    def evaluate_patch(self, gt_boxes, pred_boxes, iou_thresh=0.5, single_class=False):
        """Evaluate the model outputs against the ground truth annotations.
        If single_class is True, convert all classes to a single class.
        """
        working_gt_boxes = copy.deepcopy(gt_boxes)
        working_pred_boxes = copy.deepcopy(pred_boxes)

        if single_class:
            for box in working_gt_boxes:
                box["class_id"] = 0
            for box in working_pred_boxes:
                box["class_id"] = 0

        matches, misclassifications, false_positives, false_negatives = (
            self.match_predictions(working_gt_boxes, working_pred_boxes, iou_thresh)
        )
        confusion_matrix = self.compute_confusion_matrix(
            matches,
            misclassifications,
            false_positives,
            false_negatives,
            working_gt_boxes,
            working_pred_boxes,
            single_class=single_class,
        )
        normalized_confusion_matrix = self.compute_confusion_matrix(
            matches,
            misclassifications,
            false_positives,
            false_negatives,
            working_gt_boxes,
            working_pred_boxes,
            normalize=True,
            single_class=single_class,
        )
        # metrics = self.calculate_metrics(confusion_matrix, single_class=single_class)

        return {
            "matches": matches,
            "misclassifications": misclassifications,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            # "metrics": metrics,
            "confusion_matrix": confusion_matrix,
            "normalized_confusion_matrix": normalized_confusion_matrix,
        }

    def evaluate_image(
        self,
        patches_gt_boxes_dict,
        patches_pred_boxes_dict,
        iou_thresh=0.5,
        single_class=False,
    ):
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
        matrix_size = 2 if single_class else (self.num_classes + 1)
        global_conf_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        detailed_results = {}

        # Evaluate each patch using evaluate_patch function and aggregate the results
        for patch_key in patches_gt_boxes_dict.keys():
            patch_result = self.evaluate_patch(
                patches_gt_boxes_dict[patch_key],
                patches_pred_boxes_dict[patch_key],
                iou_thresh,
                single_class,
            )
            global_conf_matrix += patch_result["confusion_matrix"]
            detailed_results[patch_key] = patch_result

        # Normalize the global confusion matrix
        normalized_conf_matrix = self.normalize_confusion_matrix(global_conf_matrix)

        # Calculate metrics based on the global confusion matrix
        # metrics = self.calculate_metrics(global_conf_matrix, single_class=single_class)

        return {
            "patch_results": detailed_results,
            # "metrics": metrics,
            "confusion_matrix": global_conf_matrix,
            "normalized_confusion_matrix": normalized_conf_matrix,
        }

    def evaluate_dataset(
        self,
        dataset_gt_boxes_dict,
        dataset_pred_boxes_dict,
        iou_thresh=0.5,
        single_class=False,
    ):
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
        matrix_size = 2 if single_class else (self.num_classes + 1)
        dataset_conf_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        detailed_dataset_results = {}

        # Iterate over each image in the dataset
        for image_number in dataset_gt_boxes_dict.keys():

            # Evaluate each image using the evaluate_image function
            image_results = self.evaluate_image(
                dataset_gt_boxes_dict[image_number],
                dataset_pred_boxes_dict[image_number],
                iou_thresh,
                single_class,
            )
            dataset_conf_matrix += image_results["confusion_matrix"]
            detailed_dataset_results[image_number] = (
                image_results  # Store results for each image
            )

        normalized_conf_matrix = self.normalize_confusion_matrix(dataset_conf_matrix)
        metrics = self.calculate_metrics(
            confusion_matrix=dataset_conf_matrix,
            single_class=single_class,
        )

        return {
            "image_results": detailed_dataset_results,
            "metrics": metrics,
            "confusion_matrix": dataset_conf_matrix,
            "normalized_confusion_matrix": normalized_conf_matrix,
        }

    # ------------- Plot functions -------------
    @staticmethod
    def plot_confusion_matrix(cm, class_names=["Clear", "Cloudy", "Amber"]):
        """Plot the confusion matrix."""
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="0.2f" if cm.dtype == np.float32 else "d",
            cmap="Blues",
            xticklabels=class_names + ["Background"],
            yticklabels=class_names + ["Background"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
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
        class_labels = {0: "clear", 1: "cloudy", 2: "amber"}

        # Select appropriate colors
        colors = {0: "grey", 1: "white", 2: "orange"}
        linestyle = "dashed" if is_ground_truth else "solid"

        # Plot boxes
        for box in boxes:
            if class_label is None or box["class_id"] == class_label:
                x_min, y_min, x_max, y_max = box["bbox"]
                cls_id = box["class_id"]
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1.5,
                    edgecolor=colors[cls_id],
                    facecolor="none",
                    linestyle=linestyle,
                    label=class_labels[cls_id],
                )
                ax.add_patch(rect)

        # Set legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Removing duplicates in legend
        ax.legend(by_label.values(), by_label.keys())

        plt.show()
