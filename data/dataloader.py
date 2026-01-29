"""
DataLoader utilities for PLVLDet.
Implements distributed data loading, batch collation, and dataloader construction.
"""

import os
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.utils.data.dataloader import default_collate
from typing import Dict, List, Tuple, Optional, Any, Iterator

from .dataset import PLVLDataset, VLDetectionDataset


class InfiniteSampler(Sampler):
    """
    Wraps a sampler to yield indices indefinitely.
    Used for training with infinite iteration count.
    """
    
    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1
    ):
        self._size = size
        self._shuffle = shuffle
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
    
    def __iter__(self) -> Iterator[int]:
        start = self._rank
        yield from self._infinite_indices()
    
    def _infinite_indices(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()[self._rank::self._world_size]
            else:
                yield from torch.arange(self._size).tolist()[self._rank::self._world_size]


class AspectRatioBatchSampler(Sampler):
    """
    Batch sampler that groups samples with similar aspect ratios.
    Reduces padding and improves training efficiency.
    """
    
    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
        aspect_ratios: Optional[List[float]] = None
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.aspect_ratios = aspect_ratios
        
        if aspect_ratios is not None:
            # Group by aspect ratio bins
            bins = [0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0]
            self.groups = np.digitize(aspect_ratios, bins)
        else:
            self.groups = None
    
    def __iter__(self) -> Iterator[List[int]]:
        if self.groups is not None:
            # Aspect ratio grouping
            buckets: Dict[int, List[int]] = {}
            for idx in self.sampler:
                group = self.groups[idx]
                if group not in buckets:
                    buckets[group] = []
                buckets[group].append(idx)
                
                if len(buckets[group]) == self.batch_size:
                    yield buckets[group]
                    buckets[group] = []
            
            # Yield remaining samples
            remaining = []
            for bucket in buckets.values():
                remaining.extend(bucket)
            
            if len(remaining) > 0:
                if not self.drop_last:
                    yield remaining
        else:
            # Regular batching
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


def detection_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for object detection.
    Handles variable number of boxes per image.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch dictionary
    """
    images = []
    boxes_list = []
    labels_list = []
    img_metas = []
    text_prompts = []
    
    for sample in batch:
        images.append(sample['image'])
        boxes_list.append(sample['boxes'])
        labels_list.append(sample['labels'])
        
        if 'img_meta' in sample:
            img_metas.append(sample['img_meta'])
        
        if 'text_prompts' in sample:
            text_prompts.append(sample['text_prompts'])
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Compute max number of boxes
    max_boxes = max(len(b) for b in boxes_list) if boxes_list else 0
    batch_size = len(batch)
    
    # Pad boxes and labels
    if max_boxes > 0:
        padded_boxes = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32)
        padded_labels = torch.zeros(batch_size, max_boxes, dtype=torch.long)
        num_boxes = torch.zeros(batch_size, dtype=torch.long)
        
        for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            n = len(boxes)
            if n > 0:
                padded_boxes[i, :n] = boxes
                padded_labels[i, :n] = labels
            num_boxes[i] = n
    else:
        padded_boxes = torch.zeros(batch_size, 1, 4, dtype=torch.float32)
        padded_labels = torch.zeros(batch_size, 1, dtype=torch.long)
        num_boxes = torch.zeros(batch_size, dtype=torch.long)
    
    result = {
        'images': images,
        'boxes': padded_boxes,
        'labels': padded_labels,
        'num_boxes': num_boxes,
    }
    
    if img_metas:
        result['img_metas'] = img_metas
    
    if text_prompts:
        result['text_prompts'] = text_prompts
    
    return result


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    is_train: bool = True,
    distributed: bool = False,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True
) -> DataLoader:
    """
    Build dataloader for training or evaluation.
    
    Args:
        dataset: Dataset instance
        batch_size: Per-GPU batch size
        num_workers: Number of data loading workers
        is_train: Whether this is for training
        distributed: Whether to use distributed training
        seed: Random seed for reproducibility
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        DataLoader instance
    """
    if distributed:
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if is_train:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed
            )
    else:
        if is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last if is_train else False,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    return dataloader


def build_train_dataloader(
    cfg: Dict[str, Any],
    category_info: Dict[str, Any],
    distributed: bool = False
) -> DataLoader:
    """
    Build training dataloader from config.
    
    Args:
        cfg: Training configuration
        category_info: Category definitions and text prompts
        distributed: Whether to use distributed training
        
    Returns:
        Training DataLoader
    """
    dataset = VLDetectionDataset(
        img_dir=cfg['data']['train_img_dir'],
        label_dir=cfg['data']['train_label_dir'],
        category_info=category_info,
        img_size=cfg['data'].get('img_size', 1000),
        mosaic_prob=cfg['augmentation'].get('mosaic_prob', 0.5),
        mixup_prob=cfg['augmentation'].get('mixup_prob', 0.1),
        is_train=True,
        cache_images=cfg['data'].get('cache_images', False)
    )
    
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=cfg['training']['batch_size_per_gpu'],
        num_workers=cfg['data'].get('num_workers', 4),
        is_train=True,
        distributed=distributed,
        seed=cfg.get('seed', 42),
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def build_val_dataloader(
    cfg: Dict[str, Any],
    category_info: Dict[str, Any],
    distributed: bool = False
) -> DataLoader:
    """
    Build validation dataloader from config.
    
    Args:
        cfg: Configuration
        category_info: Category definitions and text prompts
        distributed: Whether to use distributed training
        
    Returns:
        Validation DataLoader
    """
    dataset = VLDetectionDataset(
        img_dir=cfg['data']['val_img_dir'],
        label_dir=cfg['data']['val_label_dir'],
        category_info=category_info,
        img_size=cfg['data'].get('img_size', 1000),
        mosaic_prob=0.0,
        mixup_prob=0.0,
        is_train=False,
        cache_images=False
    )
    
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=cfg['training']['batch_size_per_gpu'],
        num_workers=cfg['data'].get('num_workers', 4),
        is_train=False,
        distributed=distributed,
        seed=cfg.get('seed', 42),
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


class PrefetchLoader:
    """
    Dataloader wrapper that prefetches to GPU asynchronously.
    Improves training throughput by overlapping data transfer with computation.
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.loader = loader
        self.device = device
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        
        for batch in self.loader:
            with torch.cuda.stream(stream):
                # Move to GPU
                batch['images'] = batch['images'].to(self.device, non_blocking=True)
                batch['boxes'] = batch['boxes'].to(self.device, non_blocking=True)
                batch['labels'] = batch['labels'].to(self.device, non_blocking=True)
                batch['num_boxes'] = batch['num_boxes'].to(self.device, non_blocking=True)
            
            if not first:
                yield batch_prev
            else:
                first = False
            
            torch.cuda.current_stream().wait_stream(stream)
            batch_prev = batch
        
        yield batch_prev
    
    def __len__(self) -> int:
        return len(self.loader)
    
    @property
    def sampler(self):
        return self.loader.sampler


def worker_init_fn(worker_id: int):
    """
    Initialize random seed for each worker.
    Ensures different random augmentations across workers.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
