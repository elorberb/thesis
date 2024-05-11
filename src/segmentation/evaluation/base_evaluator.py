from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    @abstractmethod
    def load_model(self):
        """Abstract method to load the model."""
        pass

    @abstractmethod
    def load_data(self):
        """Abstract method to load the dataset."""
        pass

    def iou(self, boxA, boxB):
        """Calculate the Intersection over Union (IoU) between two boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def calculate_true_false_positives(self, ground_truths, predictions):
        """Calculate true positives and false positives based on IoU threshold."""
        true_positives = 0
        false_positives = 0
        matched_gts = set()

        for pred_box in predictions:
            best_iou = 0
            best_gt_index = -1

            for gt_index, gt_box in enumerate(ground_truths):
                if gt_index in matched_gts:
                    continue

                current_iou = self.iou(pred_box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_index = gt_index

            if best_iou >= self.iou_threshold:
                true_positives += 1
                matched_gts.add(best_gt_index)
            else:
                false_positives += 1

        return true_positives, false_positives, len(ground_truths) - len(matched_gts)

    def calculate_precision(self, true_positives, false_positives):
        """Calculate precision from true positives and false positives."""
        if true_positives + false_positives == 0:
            return 0
        return true_positives / (true_positives + false_positives)

    def calculate_recall(self, true_positives, false_negatives):
        """Calculate recall from true positives and false negatives."""
        if true_positives + false_negatives == 0:
            return 0
        return true_positives / (true_positives + false_negatives)

    def evaluate(self, ground_truths, predictions):
        """Evaluate the model's performance by calculating precision and recall."""
        true_positives, false_positives, false_negatives = self.calculate_true_false_positives(ground_truths, predictions)
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)

        return {"precision": precision, "recall": recall}

    @abstractmethod
    def load_data_for_patch(self, patch):
        """Abstract method to load data for a specific patch."""
        pass
