"""
Evaluation metrics for PLVLDet.
Implements mAP calculation and confusion matrix for object detection.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings


def compute_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes using NumPy.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        iou: Array of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    iou = inter / (area1[:, None] + area2 - inter + 1e-7)
    
    return iou


def compute_ap(
    recall: np.ndarray,
    precision: np.ndarray,
    method: str = 'interp101'
) -> float:
    """
    Compute Average Precision given recall and precision curves.
    
    Args:
        recall: Recall values at different thresholds
        precision: Precision values at different thresholds
        method: 'interp101' for 101-point interpolation (COCO style),
                'interp11' for 11-point interpolation (VOC 2007),
                'area' for area under curve
                
    Returns:
        ap: Average Precision value
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    if method == 'interp101':
        # 101-point interpolation (COCO)
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    elif method == 'interp11':
        # 11-point interpolation (VOC 2007)
        x = np.linspace(0, 1, 11)
        ap = np.mean(np.interp(x, mrec, mpre))
    else:
        # Area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return float(ap)


def compute_ap_per_class(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05),
    num_classes: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute AP for each class across multiple IoU thresholds.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        targets: List of dicts with 'boxes', 'labels'
        iou_thresholds: Array of IoU thresholds
        num_classes: Number of classes
        
    Returns:
        ap_per_class: Array of shape (num_classes, num_iou_thresholds)
        precision: Array of shape (num_classes,)
        recall: Array of shape (num_classes,)
    """
    # Gather all predictions and targets per class
    stats_per_class = defaultdict(lambda: {'tp': [], 'conf': [], 'n_gt': 0})
    
    for pred, gt in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Process each class
        for cls in range(num_classes):
            # Ground truth for this class
            gt_mask = gt_labels == cls
            gt_cls_boxes = gt_boxes[gt_mask]
            n_gt = len(gt_cls_boxes)
            
            stats_per_class[cls]['n_gt'] += n_gt
            
            # Predictions for this class
            pred_mask = pred_labels == cls
            pred_cls_boxes = pred_boxes[pred_mask]
            pred_cls_scores = pred_scores[pred_mask]
            
            if len(pred_cls_boxes) == 0:
                continue
            
            if n_gt == 0:
                # All predictions are false positives
                stats_per_class[cls]['tp'].extend([np.zeros(len(iou_thresholds))] * len(pred_cls_boxes))
                stats_per_class[cls]['conf'].extend(pred_cls_scores.tolist())
                continue
            
            # Compute IoU
            iou = compute_iou_np(pred_cls_boxes, gt_cls_boxes)
            
            # Match predictions to ground truth
            gt_matched = np.zeros((len(iou_thresholds), n_gt), dtype=bool)
            
            # Sort predictions by confidence
            sorted_idx = np.argsort(-pred_cls_scores)
            
            for idx in sorted_idx:
                ious = iou[idx]
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]
                
                tp = np.zeros(len(iou_thresholds))
                for t, thresh in enumerate(iou_thresholds):
                    if max_iou >= thresh and not gt_matched[t, max_iou_idx]:
                        tp[t] = 1
                        gt_matched[t, max_iou_idx] = True
                
                stats_per_class[cls]['tp'].append(tp)
                stats_per_class[cls]['conf'].append(pred_cls_scores[idx])
    
    # Compute AP per class
    ap_per_class = np.zeros((num_classes, len(iou_thresholds)))
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    
    for cls in range(num_classes):
        stats = stats_per_class[cls]
        n_gt = stats['n_gt']
        
        if n_gt == 0 and len(stats['tp']) == 0:
            continue
        
        if len(stats['tp']) == 0:
            # No predictions
            continue
        
        # Sort by confidence
        tp = np.array(stats['tp'])  # (N, num_thresholds)
        conf = np.array(stats['conf'])
        sorted_idx = np.argsort(-conf)
        tp = tp[sorted_idx]
        
        # Compute cumulative TP and FP
        tp_cum = np.cumsum(tp, axis=0)
        fp_cum = np.cumsum(1 - tp, axis=0)
        
        # Compute precision and recall
        recall = tp_cum / (n_gt + 1e-7)
        precision = tp_cum / (tp_cum + fp_cum + 1e-7)
        
        # Compute AP for each IoU threshold
        for t in range(len(iou_thresholds)):
            ap_per_class[cls, t] = compute_ap(recall[:, t], precision[:, t])
        
        # Store final precision/recall at first IoU threshold
        if len(recall) > 0:
            recall_per_class[cls] = recall[-1, 0]
            precision_per_class[cls] = precision[-1, 0]
    
    return ap_per_class, precision_per_class, recall_per_class


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 32
) -> Dict[str, float]:
    """
    Compute mean Average Precision.
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP metrics
    """
    iou_thresholds = np.array([iou_threshold])
    ap_per_class, precision, recall = compute_ap_per_class(
        predictions, targets, iou_thresholds, num_classes
    )
    
    # Filter classes with ground truth
    valid_classes = []
    for pred, gt in zip(predictions, targets):
        valid_classes.extend(gt['labels'].tolist())
    valid_classes = list(set(valid_classes))
    
    if len(valid_classes) == 0:
        return {
            'mAP': 0.0,
            'mAP50': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    ap = ap_per_class[:, 0]
    mAP = np.mean(ap[valid_classes]) if valid_classes else 0.0
    
    return {
        'mAP': float(mAP),
        'mAP50': float(mAP),
        'precision': float(np.mean(precision[valid_classes])) if valid_classes else 0.0,
        'recall': float(np.mean(recall[valid_classes])) if valid_classes else 0.0,
        'ap_per_class': ap.tolist()
    }


def compute_map_coco(
    predictions: List[Dict],
    targets: List[Dict],
    num_classes: int = 32
) -> Dict[str, float]:
    """
    Compute COCO-style mAP (averaged over IoU 0.5:0.95).
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP metrics
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_per_class, precision, recall = compute_ap_per_class(
        predictions, targets, iou_thresholds, num_classes
    )
    
    # Filter valid classes
    valid_classes = []
    for gt in targets:
        valid_classes.extend(gt['labels'].tolist())
    valid_classes = list(set(valid_classes))
    
    if len(valid_classes) == 0:
        return {
            'mAP': 0.0,
            'mAP50': 0.0,
            'mAP75': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    # mAP at different IoU thresholds
    mAP50 = float(np.mean(ap_per_class[valid_classes, 0]))
    mAP75 = float(np.mean(ap_per_class[valid_classes, 5]))  # IoU=0.75
    mAP = float(np.mean(ap_per_class[valid_classes]))  # Average over all thresholds
    
    return {
        'mAP': mAP,
        'mAP50': mAP50,
        'mAP75': mAP75,
        'precision': float(np.mean(precision[valid_classes])),
        'recall': float(np.mean(recall[valid_classes])),
        'ap_per_class': ap_per_class.mean(axis=1).tolist()
    }


def evaluate_detection(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 32,
    conf_thresh: float = 0.001,
    iou_thresh: float = 0.65
) -> Dict[str, Any]:
    """
    Evaluate detection model on a dataset.
    
    Args:
        model: Detection model
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        num_classes: Number of classes
        conf_thresh: Confidence threshold
        iou_thresh: NMS IoU threshold
        
    Returns:
        Evaluation results dictionary
    """
    from .postprocess import non_max_suppression
    
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            num_boxes = batch['num_boxes']
            
            # Forward pass
            outputs = model(images)
            
            # Apply NMS
            detections = non_max_suppression(
                outputs['predictions'],
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh
            )
            
            # Collect predictions and targets
            batch_size = images.shape[0]
            for i in range(batch_size):
                det = detections[i]
                
                pred_dict = {
                    'boxes': det[:, :4].cpu().numpy(),
                    'scores': det[:, 4].cpu().numpy(),
                    'labels': det[:, 5].cpu().numpy().astype(int)
                }
                predictions.append(pred_dict)
                
                n = num_boxes[i].item()
                target_dict = {
                    'boxes': gt_boxes[i, :n].cpu().numpy(),
                    'labels': gt_labels[i, :n].cpu().numpy().astype(int)
                }
                targets.append(target_dict)
    
    # Compute metrics
    results = compute_map_coco(predictions, targets, num_classes)
    
    return results


class ConfusionMatrix:
    """
    Confusion matrix for object detection.
    """
    
    def __init__(self, num_classes: int, conf_thresh: float = 0.25, iou_thresh: float = 0.5):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
    
    def process_batch(
        self,
        detections: np.ndarray,
        labels: np.ndarray
    ):
        """
        Process a batch of detections and labels.
        
        Args:
            detections: Array of shape (N, 6) as [x1, y1, x2, y2, conf, class]
            labels: Array of shape (M, 5) as [class, x1, y1, x2, y2]
        """
        if detections is None:
            # No detections, all ground truth are FN
            for gt_class in labels[:, 0].astype(int):
                self.matrix[self.num_classes, gt_class] += 1
            return
        
        # Filter by confidence
        detections = detections[detections[:, 4] > self.conf_thresh]
        
        gt_classes = labels[:, 0].astype(int)
        det_classes = detections[:, 5].astype(int)
        
        if len(labels) == 0:
            # No ground truth, all detections are FP
            for det_class in det_classes:
                self.matrix[det_class, self.num_classes] += 1
            return
        
        # Compute IoU
        iou = compute_iou_np(detections[:, :4], labels[:, 1:5])
        
        # Match detections to ground truth
        x = np.where(iou > self.iou_thresh)
        
        if x[0].shape[0]:
            matches = np.concatenate(
                [x[0][:, None], x[1][:, None], iou[x[0], x[1]][:, None]], axis=1
            )
            
            if matches.shape[0] > 1:
                # Sort by IoU descending
                matches = matches[matches[:, 2].argsort()[::-1]]
                # Remove duplicate detections
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # Remove duplicate ground truths
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            
            matched_det = matches[:, 0].astype(int)
            matched_gt = matches[:, 1].astype(int)
        else:
            matched_det = np.array([], dtype=int)
            matched_gt = np.array([], dtype=int)
        
        # Count TP, FP, FN
        for i, det_class in enumerate(det_classes):
            if i in matched_det:
                gt_idx = matched_gt[matched_det == i][0]
                gt_class = gt_classes[gt_idx]
                self.matrix[det_class, gt_class] += 1  # TP or misclassification
            else:
                self.matrix[det_class, self.num_classes] += 1  # FP (background)
        
        for i, gt_class in enumerate(gt_classes):
            if i not in matched_gt:
                self.matrix[self.num_classes, gt_class] += 1  # FN
    
    def get_matrix(self) -> np.ndarray:
        """Return the confusion matrix."""
        return self.matrix
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute precision, recall, and F1 from confusion matrix."""
        tp = np.diag(self.matrix)[:-1]  # Exclude background
        fp = self.matrix[:-1, -1]  # FP = predicted as class but is background
        fn = self.matrix[-1, :-1]  # FN = actual class but predicted as background
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'precision': float(np.mean(precision)),
            'recall': float(np.mean(recall)),
            'f1': float(np.mean(f1)),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist()
        }
    
    def reset(self):
        """Reset the confusion matrix."""
        self.matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
