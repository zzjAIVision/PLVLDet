"""
Post-processing utilities for PLVLDet.
Includes Non-Maximum Suppression, box decoding, and box format conversions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format
        
    Returns:
        iou: Tensor of shape (N, M) containing pairwise IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    iou = inter / (area1[:, None] + area2 - inter + 1e-7)
    
    return iou


def box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU for batched inputs.
    
    Args:
        boxes1: Tensor of shape (B, N, 4)
        boxes2: Tensor of shape (B, M, 4)
        
    Returns:
        iou: Tensor of shape (B, N, M)
    """
    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
    
    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # (B, N, M, 2)
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # (B, N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (B, N, M, 2)
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # (B, N, M)
    
    iou = inter / (area1[:, :, None] + area2[:, None, :] - inter + 1e-7)
    
    return iou


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2) format.
    
    Args:
        boxes: Tensor of shape (..., 4)
        
    Returns:
        Converted boxes in xyxy format
    """
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy2xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height) format.
    
    Args:
        boxes: Tensor of shape (..., 4)
        
    Returns:
        Converted boxes in xywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def decode_boxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
    reg_max: int = 16,
    stride: float = 1.0
) -> torch.Tensor:
    """
    Decode predicted box deltas to bounding boxes using Distribution Focal Loss format.
    
    Args:
        anchors: Anchor points of shape (N, 2) representing (x, y) centers
        deltas: Predicted distributions of shape (N, 4 * reg_max)
        reg_max: Number of bins for distribution
        stride: Feature map stride
        
    Returns:
        boxes: Decoded boxes in xyxy format, shape (N, 4)
    """
    N = anchors.shape[0]
    
    # Reshape deltas to (N, 4, reg_max)
    deltas = deltas.view(N, 4, reg_max)
    
    # Apply softmax to get distribution
    deltas = F.softmax(deltas, dim=-1)
    
    # Compute expected value using integral
    proj = torch.arange(reg_max, dtype=deltas.dtype, device=deltas.device)
    dist = (deltas * proj).sum(-1)  # (N, 4)
    
    # Convert to box coordinates
    # dist represents (left, top, right, bottom) distances from anchor
    lt = anchors - dist[:, :2] * stride
    rb = anchors + dist[:, 2:] * stride
    
    boxes = torch.cat([lt, rb], dim=-1)
    
    return boxes


def non_max_suppression(
    predictions: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.65,
    max_det: int = 300,
    multi_label: bool = True,
    agnostic: bool = False
) -> List[torch.Tensor]:
    """
    Perform Non-Maximum Suppression on detection predictions.
    
    Args:
        predictions: Predictions tensor of shape (batch_size, num_anchors, 5 + num_classes)
                    where each prediction is [x1, y1, x2, y2, obj_conf, cls1, cls2, ...]
        conf_thresh: Confidence threshold for filtering
        iou_thresh: IoU threshold for NMS
        max_det: Maximum number of detections per image
        multi_label: Allow multiple labels per box
        agnostic: Class-agnostic NMS
        
    Returns:
        List of detections, each tensor of shape (n, 6) as [x1, y1, x2, y2, conf, class]
    """
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[2] - 5
    
    # Settings
    max_wh = 7680  # Maximum box width/height
    max_nms = 30000  # Maximum number of boxes into NMS
    
    output = [torch.zeros((0, 6), device=predictions.device)] * batch_size
    
    for idx in range(batch_size):
        pred = predictions[idx]  # (num_anchors, 5 + num_classes)
        
        # Filter by confidence
        obj_conf = pred[:, 4]
        mask = obj_conf > conf_thresh
        pred = pred[mask]
        
        if pred.shape[0] == 0:
            continue
        
        # Compute combined confidence
        if num_classes == 1:
            # Single class
            pred[:, 5:] = pred[:, 4:5]
        else:
            pred[:, 5:] = pred[:, 4:5] * pred[:, 5:]  # conf = obj * cls
        
        # Box coordinates
        boxes = pred[:, :4]
        
        if multi_label:
            # Multiple labels per box
            i, j = (pred[:, 5:] > conf_thresh).nonzero(as_tuple=True)
            pred = torch.cat([boxes[i], pred[i, 5 + j, None], j[:, None].float()], dim=1)
        else:
            # Best class only
            conf, j = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat([boxes, conf, j.float()], dim=1)
            pred = pred[conf.view(-1) > conf_thresh]
        
        if pred.shape[0] == 0:
            continue
        
        # Sort by confidence
        pred = pred[pred[:, 4].argsort(descending=True)[:max_nms]]
        
        # Apply NMS
        cls = pred[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pred[:, :4] + cls
        scores = pred[:, 4]
        
        keep = nms_torch(boxes, scores, iou_thresh)
        keep = keep[:max_det]
        
        output[idx] = pred[keep]
    
    return output


def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """
    Pure PyTorch implementation of NMS.
    
    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        scores: Tensor of shape (N,)
        iou_thresh: IoU threshold
        
    Returns:
        keep: Indices of kept boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        
        mask = iou <= iou_thresh
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.65,
    score_thresh: float = 0.001,
    sigma: float = 0.5,
    method: str = 'gaussian'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS implementation.
    
    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        scores: Tensor of shape (N,)
        iou_thresh: IoU threshold for linear decay
        score_thresh: Score threshold for filtering
        sigma: Gaussian sigma
        method: 'linear' or 'gaussian'
        
    Returns:
        keep: Indices of kept boxes
        new_scores: Updated scores
    """
    N = boxes.shape[0]
    indices = torch.arange(N, device=boxes.device)
    keep = []
    new_scores = scores.clone()
    
    for _ in range(N):
        max_idx = new_scores.argmax()
        max_score = new_scores[max_idx]
        
        if max_score < score_thresh:
            break
        
        keep.append(indices[max_idx].item())
        
        # Compute IoU with remaining boxes
        iou = box_iou(boxes[max_idx:max_idx+1], boxes)[0]
        
        # Update scores
        if method == 'linear':
            decay = torch.where(iou > iou_thresh, 1 - iou, torch.ones_like(iou))
        else:  # gaussian
            decay = torch.exp(-(iou ** 2) / sigma)
        
        new_scores = new_scores * decay
        new_scores[max_idx] = 0
    
    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    return keep, new_scores[keep] if len(keep) > 0 else new_scores[:0]


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thresh: float
) -> torch.Tensor:
    """
    Class-aware NMS.
    
    Args:
        boxes: Tensor of shape (N, 4)
        scores: Tensor of shape (N,)
        labels: Tensor of shape (N,)
        iou_thresh: IoU threshold
        
    Returns:
        keep: Indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Strategy: offset boxes by class
    max_coord = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    keep = nms_torch(boxes_for_nms, scores, iou_thresh)
    
    return keep


def clip_boxes(boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: Tensor of shape (..., 4) in xyxy format
        img_shape: (height, width)
        
    Returns:
        Clipped boxes
    """
    h, w = img_shape
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h)
    return boxes


def scale_boxes(
    boxes: torch.Tensor,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None
) -> torch.Tensor:
    """
    Scale boxes from one image size to another.
    
    Args:
        boxes: Boxes to scale
        from_shape: Original shape (h, w)
        to_shape: Target shape (h, w)
        ratio_pad: Optional (ratio, (dw, dh)) from letterbox
        
    Returns:
        Scaled boxes
    """
    if ratio_pad is None:
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad = (
            (from_shape[1] - to_shape[1] * gain) / 2,
            (from_shape[0] - to_shape[0] * gain) / 2
        )
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    
    return clip_boxes(boxes, to_shape)
