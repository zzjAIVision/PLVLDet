"""
Dataset Classes for PLVLDet

This module implements dataset classes for:
- Loading LabelMe annotations
- Creating object-text pairs for vision-language training
- Supporting mosaic and mixup augmentation
"""

import os
import json
import random
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
import cv2


class LabelMeParser:
    """Parser for LabelMe format annotations."""
    
    def __init__(self, category_mapping: Dict[str, int]):
        """
        Initialize parser.
        
        Args:
            category_mapping: Mapping from category name to class index
        """
        self.category_mapping = category_mapping
        
    def parse(self, json_path: str) -> Dict[str, Any]:
        """
        Parse LabelMe JSON annotation file.
        
        Args:
            json_path: Path to annotation JSON file
            
        Returns:
            Dictionary with 'boxes' (N, 4) and 'labels' (N,)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        boxes = []
        labels = []
        
        for shape in data.get('shapes', []):
            if shape['shape_type'] != 'rectangle':
                continue
                
            label = shape['label']
            if label not in self.category_mapping:
                continue
                
            points = shape['points']
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_mapping[label])
            
        return {
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            'image_size': (data.get('imageHeight', 0), data.get('imageWidth', 0))
        }


class PLVLDataset(Dataset):
    """
    Power Line Vision-Language Detection Dataset.
    
    Loads images and annotations, constructs object-text pairs for
    vision-language pre-training and detection.
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        category_config: Dict,
        transform: Optional[Any] = None,
        mosaic_prob: float = 0.0,
        mixup_prob: float = 0.0,
        img_size: int = 1000
    ):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing LabelMe JSON annotations
            category_config: Category configuration with text prompts
            transform: Data augmentation transforms
            mosaic_prob: Probability of mosaic augmentation
            mixup_prob: Probability of mixup augmentation
            img_size: Target image size
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.img_size = img_size
        
        # Build category mapping
        self.category_config = category_config
        self.category_mapping = self._build_category_mapping()
        self.category_texts = self._build_category_texts()
        self.num_classes = len(self.category_mapping)
        
        # Initialize annotation parser
        self.parser = LabelMeParser(self.category_mapping)
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        print(f"Loaded {len(self.samples)} samples with {self.num_classes} categories")
        
    def _build_category_mapping(self) -> Dict[str, int]:
        """Build mapping from category name/abbr to class index."""
        mapping = {}
        categories = self.category_config.get('categories', {})
        
        for cat_name, cat_info in categories.items():
            cat_id = cat_info['id']
            abbr = cat_info.get('abbr', cat_name)
            
            # Map both full name and abbreviation
            mapping[cat_name] = cat_id
            mapping[abbr] = cat_id
            mapping[cat_info['name']] = cat_id
            
        return mapping
    
    def _build_category_texts(self) -> Dict[str, List[str]]:
        """Build category text prompts dictionary."""
        texts = {}
        categories = self.category_config.get('categories', {})
        
        for cat_name, cat_info in categories.items():
            cat_id = cat_info['id']
            prompts = cat_info.get('text_prompts', [cat_info['name']])
            texts[str(cat_id)] = prompts
            
        return texts
    
    def _build_sample_list(self) -> List[Tuple[str, str]]:
        """Build list of (image_path, annotation_path) tuples."""
        samples = []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for filename in os.listdir(self.image_dir):
            name, ext = os.path.splitext(filename)
            if ext.lower() not in image_extensions:
                continue
                
            image_path = os.path.join(self.image_dir, filename)
            label_path = os.path.join(self.label_dir, name + '.json')
            
            if os.path.exists(label_path):
                samples.append((image_path, label_path))
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def load_image(self, idx: int) -> np.ndarray:
        """Load image at index."""
        image_path = self.samples[idx][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_annotation(self, idx: int) -> Dict:
        """Load annotation at index."""
        label_path = self.samples[idx][1]
        return self.parser.parse(label_path)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns dictionary with:
        - image: Tensor (3, H, W)
        - boxes: Tensor (N, 4) in xyxy format
        - labels: Tensor (N,) class indices
        - text_labels: List of text descriptions
        """
        # Decide augmentation type
        use_mosaic = random.random() < self.mosaic_prob
        use_mixup = random.random() < self.mixup_prob and not use_mosaic
        
        if use_mosaic:
            return self._load_mosaic(idx)
        elif use_mixup:
            return self._load_mixup(idx)
        else:
            return self._load_single(idx)
    
    def _load_single(self, idx: int) -> Dict:
        """Load a single sample without mosaic/mixup."""
        image = self.load_image(idx)
        annotation = self.load_annotation(idx)
        
        boxes = annotation['boxes']
        labels = annotation['labels']
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(boxes) if len(boxes) > 0 else torch.zeros((0, 4))
        labels = torch.from_numpy(labels) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)
        
        # Get text labels for each object
        text_labels = []
        for label in labels.tolist():
            prompts = self.category_texts.get(str(label), ['unknown'])
            text_labels.append(random.choice(prompts))
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'text_labels': text_labels,
            'image_id': idx
        }
    
    def _load_mosaic(self, idx: int) -> Dict:
        """Load 4 images in mosaic pattern."""
        indices = [idx] + random.choices(range(len(self)), k=3)
        
        # Create mosaic image
        s = self.img_size
        mosaic_image = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
        
        all_boxes = []
        all_labels = []
        
        # Mosaic center
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        
        for i, index in enumerate(indices):
            image = self.load_image(index)
            annotation = self.load_annotation(index)
            h, w = image.shape[:2]
            
            # Placement coordinates
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                
            # Place image
            mosaic_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            
            # Adjust boxes
            padw = x1a - x1b
            padh = y1a - y1b
            
            boxes = annotation['boxes'].copy()
            if len(boxes) > 0:
                boxes[:, [0, 2]] += padw
                boxes[:, [1, 3]] += padh
                all_boxes.append(boxes)
                all_labels.append(annotation['labels'])
                
        # Combine annotations
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            
        # Clip boxes to mosaic bounds
        boxes = np.clip(boxes, 0, s * 2)
        
        # Resize mosaic to target size
        mosaic_image = cv2.resize(mosaic_image, (s, s))
        boxes = boxes * (s / (s * 2))
        
        # Filter valid boxes
        valid_idx = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes = boxes[valid_idx]
        labels = labels[valid_idx]
        
        # Apply remaining transforms
        if self.transform is not None:
            transformed = self.transform(
                image=mosaic_image,
                bboxes=boxes,
                labels=labels
            )
            mosaic_image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32) if transformed['bboxes'] else np.zeros((0, 4))
            labels = np.array(transformed['labels'], dtype=np.int64) if transformed['labels'] else np.zeros((0,))
        
        # Convert to tensors
        image = torch.from_numpy(mosaic_image).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(boxes) if len(boxes) > 0 else torch.zeros((0, 4))
        labels = torch.from_numpy(labels) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)
        
        # Get text labels
        text_labels = []
        for label in labels.tolist():
            prompts = self.category_texts.get(str(int(label)), ['unknown'])
            text_labels.append(random.choice(prompts))
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'text_labels': text_labels,
            'image_id': idx
        }
    
    def _load_mixup(self, idx: int) -> Dict:
        """Load two images with mixup augmentation."""
        sample1 = self._load_single(idx)
        sample2 = self._load_single(random.randint(0, len(self) - 1))
        
        # Mixup ratio
        alpha = 0.5
        
        # Blend images
        image = alpha * sample1['image'] + (1 - alpha) * sample2['image']
        
        # Combine boxes and labels
        boxes = torch.cat([sample1['boxes'], sample2['boxes']], dim=0)
        labels = torch.cat([sample1['labels'], sample2['labels']], dim=0)
        text_labels = sample1['text_labels'] + sample2['text_labels']
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'text_labels': text_labels,
            'image_id': idx
        }


class VLDetectionDataset(PLVLDataset):
    """
    Vision-Language Detection Dataset with enhanced text handling.
    
    Extends PLVLDataset with additional features for:
    - Dynamic text prompt sampling during training
    - Negative category sampling for contrastive learning
    - Support for hierarchical category labels
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        category_config: Dict,
        transform: Optional[Any] = None,
        mosaic_prob: float = 0.0,
        mixup_prob: float = 0.0,
        img_size: int = 1000,
        negative_samples: int = 10
    ):
        super().__init__(
            image_dir, label_dir, category_config,
            transform, mosaic_prob, mixup_prob, img_size
        )
        self.negative_samples = negative_samples
        
        # Build hierarchy mapping
        self.hierarchy = category_config.get('hierarchy', {})
        self.parent_to_children = {}
        self.child_to_parent = {}
        
        for parent, info in self.hierarchy.items():
            children = info.get('children', [])
            self.parent_to_children[parent] = children
            for child in children:
                self.child_to_parent[child] = parent
                
    def get_negative_categories(self, positive_labels: List[int]) -> List[int]:
        """
        Sample negative categories for contrastive learning.
        
        Args:
            positive_labels: List of positive category indices
            
        Returns:
            List of negative category indices
        """
        positive_set = set(positive_labels)
        all_categories = list(range(self.num_classes))
        negative_pool = [c for c in all_categories if c not in positive_set]
        
        num_neg = min(self.negative_samples, len(negative_pool))
        return random.sample(negative_pool, num_neg)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample with negative categories for contrastive learning."""
        sample = super().__getitem__(idx)
        
        # Add negative categories
        positive_labels = sample['labels'].tolist()
        negative_labels = self.get_negative_categories(positive_labels)
        
        # Get text prompts for negative categories
        negative_texts = []
        for label in negative_labels:
            prompts = self.category_texts.get(str(label), ['unknown'])
            negative_texts.append(random.choice(prompts))
            
        sample['negative_labels'] = torch.tensor(negative_labels, dtype=torch.long)
        sample['negative_texts'] = negative_texts
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    
    Handles variable-length boxes per image.
    """
    images = torch.stack([item['image'] for item in batch])
    
    targets = []
    for item in batch:
        targets.append({
            'boxes': item['boxes'],
            'labels': item['labels'],
            'text_labels': item.get('text_labels', []),
            'image_id': item.get('image_id', 0)
        })
    
    return {
        'images': images,
        'targets': targets
    }
