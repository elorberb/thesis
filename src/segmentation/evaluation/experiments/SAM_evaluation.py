import numpy as np


def calculate_precision_recall_per_segment(gt_dict, pred_dict, iou_threshold):
    TPs = 0  # True positives
    FPs = 0  # False positives
    FNs = 0  # False negatives
    iou_values = []  # list to store IOU values

    gt_segments = np.unique(
        gt_dict["instance_bitmap"]
    )  # List of unique segments in ground truth
    gt_segments = gt_segments[gt_segments != 0]  # Exclude the background (0)

    matched_gt_segments = set()  # Track which gt segments have been matched

    # Iterate over each predicted segment
    for segment in pred_dict["mask"]:
        pred_mask = segment["segmentation"]
        max_iou = 0
        best_match_gt_seg = None

        for seg in gt_segments:
            if seg not in matched_gt_segments:  # Only consider unmatched gt segments
                gt_mask = (
                    gt_dict["instance_bitmap"] == seg
                )  # mask for current gt segment
                intersection = np.logical_and(gt_mask, pred_mask)
                union = np.logical_or(gt_mask, pred_mask)
                iou = np.sum(intersection) / np.sum(union)

                if iou > max_iou:
                    max_iou = iou
                    best_match_gt_seg = seg

        if max_iou >= iou_threshold and best_match_gt_seg is not None:
            TPs += 1  # We have a true positive
            matched_gt_segments.add(
                best_match_gt_seg
            )  # This gt segment has been matched
        else:
            FPs += 1  # We have a false positive

        iou_values.append(max_iou)  # Add the max IOU to the list

    # Calculate false negatives by subtracting the number of matched gt segments from total gt segments
    FNs = len(gt_segments) - len(matched_gt_segments)

    precision = TPs / (TPs + FPs) if TPs + FPs > 0 else 0
    recall = TPs / (TPs + FNs) if TPs + FNs > 0 else 0

    return precision, recall, (TPs, FPs, FNs), iou_values


def calculate_average_precision_recall(ground_truth, SAM_pred, iou_threshold):
    precision_values = []
    recall_values = []
    total_TPs = 0
    total_FPs = 0
    total_FNs = 0
    all_iou_values = []
    num_images = len(ground_truth)

    for name in ground_truth:
        (
            precision,
            recall,
            (TPs, FPs, FNs),
            iou_values,
        ) = calculate_precision_recall_per_segment(
            ground_truth[name], SAM_pred[name], iou_threshold
        )
        precision_values.append(precision)
        recall_values.append(recall)
        total_TPs += TPs
        total_FPs += FPs
        total_FNs += FNs
        all_iou_values.extend(iou_values)

    average_precision = np.mean(precision_values)
    average_recall = np.mean(recall_values)
    std_dev_precision = np.std(precision_values)
    std_dev_recall = np.std(recall_values)

    return (
        average_precision,
        average_recall,
        std_dev_precision,
        std_dev_recall,
        (total_TPs, total_FPs, total_FNs),
        all_iou_values,
    )
