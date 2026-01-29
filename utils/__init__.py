"""
PLVLDet Utilities Package

This package provides:
- Post-processing utilities (NMS, box decoding)
- Evaluation metrics (mAP calculation)
- Visualization tools
- Checkpoint management
- Logging utilities
"""

from .postprocess import (
    non_max_suppression,
    decode_boxes,
    box_iou,
    xywh2xyxy,
    xyxy2xywh
)
from .metrics import (
    compute_ap,
    compute_map,
    evaluate_detection,
    ConfusionMatrix
)
from .visualization import (
    visualize_detections,
    plot_training_curves,
    save_detection_results
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    resume_from_checkpoint
)
from .logger import (
    setup_logger,
    get_logger
)

__all__ = [
    'non_max_suppression',
    'decode_boxes',
    'box_iou',
    'xywh2xyxy',
    'xyxy2xywh',
    'compute_ap',
    'compute_map',
    'evaluate_detection',
    'ConfusionMatrix',
    'visualize_detections',
    'plot_training_curves',
    'save_detection_results',
    'save_checkpoint',
    'load_checkpoint',
    'resume_from_checkpoint',
    'setup_logger',
    'get_logger'
]
