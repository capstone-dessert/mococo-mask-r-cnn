import numpy as np
import torch


def iou(preds, gts):
    """
    Calculates the Intersection over Union (IoU) for each pair of predicted and ground truth boxes.
    
    Args:
    preds (Tensor): Predicted bounding boxes, shape (N, 4)
    gts (Tensor): Ground truth bounding boxes, shape (M, 4)
    
    Returns:
    Tensor: IoU scores, shape (N, M)
    """
    assert preds.shape[1] == 4 and gts.shape[1] == 4, "Boxes must be in the format (xmin, ymin, xmax, ymax)"

    # Calculate areas of bounding boxes
    area_preds = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    
    # Calculate intersections
    lt = torch.max(preds[:, None, :2], gts[:, :2])  # Left-top corners. Shape (N, M, 2)
    rb = torch.min(preds[:, None, 2:], gts[:, 2:])  # Right-bottom corners. Shape (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # Width-height. Shape (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # Intersection area. Shape (N, M)

    # Calculate unions
    union = area_preds[:, None] + area_gts - inter  # Shape (N, M)

    iou = inter / union
    return iou



def precision_recall(iou_scores, threshold=0.5):
    """
    Calculates precision and recall based on IoU scores and a given threshold.
    
    Args:
    - iou_scores (tensor): IoU scores for all predictions with respect to a single ground truth.
    - threshold (float): IoU threshold to consider a prediction as a true positive.
    
    Returns:
    - float, float: precision, recall
    """
    true_positives = (iou_scores > threshold).sum().float()
    false_positives = (iou_scores <= threshold).sum().float()
    false_negatives = (iou_scores == 0).sum().float()  # Assuming non-detected are counted as zeros.

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision.item(), recall.item()


def calculate_map(predictions, ground_truths, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    aps = []
    for thresh in iou_thresholds:
        precisions = []
        recalls = []
        for pred, gt in zip(predictions, ground_truths):
            iou_score = iou(pred, gt)
            precision, recall = precision_recall(iou_score, threshold=thresh)
            precisions.append(precision)
            recalls.append(recall)
        # Calculate average precision for this threshold
        ap = np.mean(precisions)
        aps.append(ap)
    return np.mean(aps)  # Average over all thresholds for final mAP