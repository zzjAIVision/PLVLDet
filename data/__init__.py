"""
PLVLDet Data Processing Package

This package provides:
- Dataset classes for vision-language detection
- Data augmentation pipelines
- Data loading utilities
- Object-text pair construction
"""

from .dataset import PLVLDataset, VLDetectionDataset, LabelMeParser, collate_fn
from .transforms import (
    HSVAugment,
    RandomAffine,
    Mosaic,
    MixUp,
    LetterBox,
    PLVLDetTransform,
    build_train_transforms,
    build_val_transforms
)
from .dataloader import (
    build_dataloader,
    build_train_dataloader,
    build_val_dataloader,
    detection_collate_fn,
    PrefetchLoader,
    InfiniteSampler,
    AspectRatioBatchSampler
)

__all__ = [
    'PLVLDataset',
    'VLDetectionDataset', 
    'LabelMeParser',
    'collate_fn',
    'HSVAugment',
    'RandomAffine',
    'Mosaic',
    'MixUp',
    'LetterBox',
    'PLVLDetTransform',
    'build_train_transforms',
    'build_val_transforms',
    'build_dataloader',
    'build_train_dataloader',
    'build_val_dataloader',
    'detection_collate_fn',
    'PrefetchLoader',
    'InfiniteSampler',
    'AspectRatioBatchSampler'
]
