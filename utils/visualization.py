"""
Visualization utilities for PLVLDet.
Includes detection visualization, training curve plotting, and result saving.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json


# Color palette for 32 categories
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (255, 128, 0),    # Orange
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Lime
    (0, 255, 128),    # Spring green
    (128, 0, 255),    # Violet
    (0, 128, 255),    # Sky blue
    (255, 128, 128),  # Light red
    (128, 255, 128),  # Light green
    (128, 128, 255),  # Light blue
    (255, 255, 128),  # Light yellow
    (255, 128, 255),  # Light magenta
    (128, 255, 255),  # Light cyan
    (192, 64, 64),    # Dark red
    (64, 192, 64),    # Forest green
    (64, 64, 192),    # Dark blue
    (192, 192, 64),   # Dark yellow
    (192, 64, 192),   # Dark magenta
    (64, 192, 192),   # Dark cyan
    (160, 160, 160),  # Gray
    (96, 96, 96),     # Dark gray
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for a class ID."""
    return COLORS[class_id % len(COLORS)]


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    conf_thresh: float = 0.25,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_labels: bool = True,
    show_scores: bool = True
) -> np.ndarray:
    """
    Draw detection boxes on an image.
    
    Args:
        image: Input image (BGR format)
        boxes: Array of shape (N, 4) in xyxy format
        labels: Array of shape (N,) with class indices
        scores: Optional array of shape (N,) with confidence scores
        class_names: Optional list of class names
        conf_thresh: Confidence threshold for display
        line_thickness: Box line thickness
        font_scale: Font scale for labels
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores
        
    Returns:
        Image with drawn detections
    """
    image = image.copy()
    
    if len(boxes) == 0:
        return image
    
    # Filter by confidence
    if scores is not None:
        mask = scores >= conf_thresh
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(int(label))
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        if show_labels or show_scores:
            # Prepare label text
            if class_names is not None:
                class_name = class_names[int(label)]
            else:
                class_name = f"cls_{int(label)}"
            
            if show_scores and scores is not None:
                label_text = f"{class_name}: {scores[i]:.2f}"
            else:
                label_text = class_name
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Position label above box, or below if near top edge
            if y1 > text_height + 5:
                text_y = y1 - 5
                bg_y1 = y1 - text_height - 10
                bg_y2 = y1
            else:
                text_y = y2 + text_height + 5
                bg_y1 = y2
                bg_y2 = y2 + text_height + 10
            
            cv2.rectangle(
                image,
                (x1, bg_y1),
                (x1 + text_width + 5, bg_y2),
                color, -1
            )
            
            # Draw label text
            cv2.putText(
                image, label_text, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA
            )
    
    return image


def visualize_comparison(
    image: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    conf_thresh: float = 0.25
) -> np.ndarray:
    """
    Visualize predictions alongside ground truth.
    
    Args:
        image: Input image
        pred_boxes: Predicted boxes
        pred_labels: Predicted labels
        gt_boxes: Ground truth boxes
        gt_labels: Ground truth labels
        pred_scores: Prediction confidence scores
        class_names: Class names list
        conf_thresh: Confidence threshold
        
    Returns:
        Image with both predictions (solid) and GT (dashed) drawn
    """
    image = image.copy()
    
    # Draw ground truth boxes (dashed)
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(int(label))
        
        # Draw dashed rectangle
        for j in range(0, int(x2 - x1), 10):
            cv2.line(image, (x1 + j, y1), (min(x1 + j + 5, x2), y1), color, 1)
            cv2.line(image, (x1 + j, y2), (min(x1 + j + 5, x2), y2), color, 1)
        for j in range(0, int(y2 - y1), 10):
            cv2.line(image, (x1, y1 + j), (x1, min(y1 + j + 5, y2)), color, 1)
            cv2.line(image, (x2, y1 + j), (x2, min(y1 + j + 5, y2)), color, 1)
    
    # Draw predictions (solid)
    image = visualize_detections(
        image, pred_boxes, pred_labels, pred_scores,
        class_names, conf_thresh, line_thickness=2,
        show_labels=True, show_scores=True
    )
    
    return image


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot training curves from logged metrics.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(values, linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = 'Training Loss'
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: Training loss values
        val_losses: Optional validation loss values
        save_path: Optional path to save the figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1.5)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_all_classes: bool = False
):
    """
    Plot Precision-Recall curves.
    
    Args:
        precisions: Array of shape (num_classes, num_points)
        recalls: Array of shape (num_classes, num_points)
        class_names: Optional class names
        save_path: Optional path to save the figure
        show_all_classes: Whether to show all classes or just mean
    """
    plt.figure(figsize=(10, 8))
    
    num_classes = len(precisions)
    
    if show_all_classes:
        for i in range(num_classes):
            label = class_names[i] if class_names else f"Class {i}"
            plt.plot(recalls[i], precisions[i], label=label, linewidth=1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # Plot mean PR curve
        mean_precision = np.mean(precisions, axis=0)
        mean_recall = np.mean(recalls, axis=0)
        plt.plot(mean_recall, mean_precision, 'b-', linewidth=2, label='Mean PR')
        
        # Add shaded area for std
        std_precision = np.std(precisions, axis=0)
        plt.fill_between(
            mean_recall,
            mean_precision - std_precision,
            mean_precision + std_precision,
            alpha=0.2
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12)
):
    """
    Plot confusion matrix.
    
    Args:
        matrix: Confusion matrix array
        class_names: Optional class names
        normalize: Whether to normalize the matrix
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / (row_sums + 1e-7)
    
    plt.figure(figsize=figsize)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    num_classes = len(matrix)
    tick_marks = np.arange(num_classes)
    
    if class_names is not None:
        labels = list(class_names) + ['Background']
        plt.xticks(tick_marks, labels, rotation=45, ha='right', fontsize=8)
        plt.yticks(tick_marks, labels, fontsize=8)
    else:
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
    
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_detection_results(
    results: List[Dict],
    save_path: str,
    format: str = 'json'
):
    """
    Save detection results to file.
    
    Args:
        results: List of detection result dictionaries
        save_path: Output file path
        format: Output format ('json', 'txt', 'coco')
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for r in results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                else:
                    sr[k] = v
            serializable_results.append(sr)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif format == 'txt':
        with open(save_path, 'w') as f:
            for r in results:
                img_id = r.get('image_id', 'unknown')
                boxes = r.get('boxes', [])
                labels = r.get('labels', [])
                scores = r.get('scores', [])
                
                for box, label, score in zip(boxes, labels, scores):
                    line = f"{img_id} {label} {score:.4f} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n"
                    f.write(line)
    
    elif format == 'coco':
        # COCO format
        coco_results = []
        for r in results:
            img_id = r.get('image_id', 0)
            boxes = r.get('boxes', [])
            labels = r.get('labels', [])
            scores = r.get('scores', [])
            
            for box, label, score in zip(boxes, labels, scores):
                # Convert xyxy to xywh
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                coco_results.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })
        
        with open(save_path, 'w') as f:
            json.dump(coco_results, f)


def create_detection_video(
    images: List[np.ndarray],
    detections: List[Dict],
    output_path: str,
    fps: int = 10,
    class_names: Optional[List[str]] = None
):
    """
    Create video from detection results.
    
    Args:
        images: List of images
        detections: List of detection dictionaries
        output_path: Output video path
        fps: Frames per second
        class_names: Optional class names
    """
    if len(images) == 0:
        return
    
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img, det in zip(images, detections):
        boxes = det.get('boxes', np.array([]))
        labels = det.get('labels', np.array([]))
        scores = det.get('scores', None)
        
        frame = visualize_detections(
            img, boxes, labels, scores, class_names,
            conf_thresh=0.25, show_labels=True, show_scores=True
        )
        writer.write(frame)
    
    writer.release()
